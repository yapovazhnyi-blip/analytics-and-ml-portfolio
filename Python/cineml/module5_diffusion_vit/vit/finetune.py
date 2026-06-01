"""
Module 5 — Vision Transformer (ViT) fine-tuning + Attention Rollout.

Workflow:
  1. Fine-tune google/vit-base-patch16-224 on TMDB poster images (genre classification)
  2. Attention Rollout: visualise which patches drive the classification

Attention Rollout Reference:
  Abnar & Zuidema (2020) — Quantifying Attention Flow in Transformers

Usage:
    python finetune.py --data-dir data/raw/tmdb/posters --epochs 10
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import mlflow

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Horror", "Documentary"]
N_CLASSES = len(GENRES)
MODEL_NAME = "google/vit-base-patch16-224"


# ── Dataset ────────────────────────────────────────────────────────────────────

class PosterDataset(Dataset):
    """
    TMDB poster images with genre labels.
    Expects: data_dir/{genre}/{movie_id}.jpg
    """

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, data_dir: Path, split: str = "train", sample_size: int | None = None):
        data_dir = Path(data_dir)   # ensure Path object even if str passed in
        self.paths, self.labels = [], []
        log.info("PosterDataset: scanning %s", data_dir)
        for label_idx, genre in enumerate(GENRES):
            genre_dir = data_dir / genre
            found = list(genre_dir.glob("*.jpg")) if genre_dir.exists() else []
            log.info("  %s: %d images (dir exists: %s)", genre, len(found), genre_dir.exists())
            for img_path in found:
                self.paths.append(img_path)
                self.labels.append(label_idx)
        log.info("PosterDataset total: %d images", len(self.paths))
        # Optional subset — useful for CPU runs or quick validation
        if sample_size is not None and sample_size < len(self.paths):
            import random
            random.seed(42)
            indices = random.sample(range(len(self.paths)), sample_size)
            self.paths  = [self.paths[i]  for i in indices]
            self.labels = [self.labels[i] for i in indices]
            log.info("PosterDataset sampled to %d images", sample_size)
        self.transform = self.TRANSFORM if split == "train" else self.VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def finetune_vit(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-4,
    sample_size: int | None = None,   # limit images per split for CPU runs
):
    """Fine-tune ViT on genre classification with MLflow tracking."""
    try:
        from transformers import ViTForImageClassification, ViTConfig
    except ImportError:
        raise ImportError("pip install transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load pre-trained ViT
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Freeze all layers except the classification head (linear probe first)
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    train_ds = PosterDataset(data_dir, split="train", sample_size=sample_size)
    val_size  = max(1, int(sample_size * 0.2)) if sample_size else None
    val_ds    = PosterDataset(data_dir, split="val", sample_size=val_size)
    # num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0)

    optimiser = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("cineml-vit-finetune")
    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size, "model": MODEL_NAME})

        for epoch in range(1, epochs + 1):
            # ── Phase 1: linear probe → unfreeze all at epoch 3 ──
            if epoch == 3:
                log.info("Unfreezing all ViT layers for full fine-tune …")
                for param in model.parameters():
                    param.requires_grad = True
                for g in optimiser.param_groups:
                    g["lr"] = lr / 10  # lower LR for backbone

            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item()
                correct += (outputs.argmax(-1) == labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            val_acc = _evaluate(model, val_loader, device)
            train_acc = correct / total

            log.info("Epoch %d | train_loss=%.4f  train_acc=%.3f  val_acc=%.3f",
                     epoch, train_loss / len(train_loader), train_acc, val_acc)
            mlflow.log_metrics(
                {"train_loss": train_loss / len(train_loader),
                 "train_acc": train_acc, "val_acc": val_acc}, step=epoch
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        mlflow.log_artifact(str(output_dir))
        log.info("Model saved to %s", output_dir)


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(pixel_values=images).logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ── Attention Rollout ─────────────────────────────────────────────────────────

class AttentionRollout:
    """
    Visualises which image patches the ViT attends to for a given classification.

    Method: multiply attention matrices across all layers (with skip connection
    accounting), producing a single 14×14 saliency map.

    Reference: Abnar & Zuidema (2020)
    """

    def __init__(self, model, discard_ratio: float = 0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self._attentions: list[torch.Tensor] = []
        self._hooks: list = []

    def _hook(self, module, input, output):
        """Capture attention weights from each ViT encoder layer."""
        # HuggingFace ViT returns tuple (context, attention_weights) when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2:
            self._attentions.append(output[1].detach().cpu())

    def register_hooks(self):
        for layer in self.model.vit.encoder.layer:
            h = layer.attention.attention.register_forward_hook(self._hook)
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.inference_mode()
    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            image_tensor: (1, 3, 224, 224) normalised image

        Returns:
            rollout: (14, 14) numpy attention map (values in [0, 1])
        """
        self._attentions.clear()
        self.register_hooks()

        self.model.eval()
        _ = self.model(pixel_values=image_tensor, output_attentions=True)
        self.remove_hooks()

        if not self._attentions:
            raise RuntimeError("No attention weights captured. Check model architecture.")

        # Rollout: A_rolled = A_L @ A_{L-1} @ … @ A_1
        result = torch.eye(self._attentions[0].shape[-1])
        for attn in self._attentions:
            # Average over heads
            attn_avg = attn.mean(dim=1)  # (B, seq_len, seq_len)
            # Add residual connection (I) — Equation 2 from Abnar & Zuidema
            attn_avg = (attn_avg + torch.eye(attn_avg.shape[-1])) / 2
            # Discard low-attention tokens
            flat = attn_avg.view(-1)
            threshold = torch.quantile(flat, self.discard_ratio)
            attn_avg = torch.where(attn_avg >= threshold, attn_avg, torch.zeros_like(attn_avg))
            attn_avg = attn_avg / (attn_avg.sum(dim=-1, keepdim=True) + 1e-9)
            result = attn_avg[0] @ result

        # CLS token (index 0) attends to patch tokens (1:)
        mask = result[0, 1:]  # (196,) for 14×14 patches
        mask = mask.reshape(14, 14).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
        return mask


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/vit_genre"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit training images per split. None = all available.")
    args = parser.parse_args()

    finetune_vit(args.data_dir, args.output_dir,
                 epochs=args.epochs,
                 batch_size=args.batch_size,
                 sample_size=args.sample_size)
