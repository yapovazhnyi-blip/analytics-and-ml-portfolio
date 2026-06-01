"""
Module 5 — DDPM Training Script.

Trains the class-conditional U-Net on:
  - MNIST (CPU, quick sanity check)
  - TMDB poster thumbnails (GPU, full training)

Tracks all experiments with MLflow: loss curves, FID, sample grids.

Usage:
    # Sanity check on MNIST (CPU, ~10min)
    python train.py --dataset mnist --epochs 20 --image-size 32

    # Full training on TMDB posters (GPU recommended)
    python train.py --dataset tmdb --epochs 200 --image-size 64
"""
import argparse
import logging
import sys
from pathlib import Path

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Repo root on sys.path ─────────────────────────────────────────────────────
# Required so that module5_diffusion_vit.diffusion.unet etc. are importable
# when running this script directly from any working directory.
_ROOT = Path(__file__).resolve().parent.parent.parent  # cineml/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from unet import UNet
from ddpm import DiffusionScheduler, compute_loss, ddpm_sample, ddim_sample

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

MODEL_DIR = _ROOT / "data" / "models"
SAMPLE_DIR = _ROOT / "data" / "samples"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset loaders ────────────────────────────────────────────────────────────

def _to_rgb(x):
    """Convert 1-channel greyscale to 3-channel RGB. Must be module-level to be picklable on Windows."""
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


def get_mnist_loader(image_size: int, batch_size: int, sample_size: int | None = None) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(_to_rgb),   # module-level function, safe to pickle
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    # num_workers=0 and pin_memory=False for Windows CPU compatibility
    ds = datasets.MNIST(str(_ROOT / "data" / "raw"), train=True, download=True, transform=transform)
    if sample_size is not None and sample_size < len(ds):
        import torch
        indices = torch.randperm(len(ds))[:sample_size]
        ds = torch.utils.data.Subset(ds, indices)
        log.info("MNIST sampled to %d images (from 60,000)", sample_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)


def get_tmdb_loader(data_dir: Path, image_size: int, batch_size: int) -> DataLoader:
    from module5_diffusion_vit.vit.finetune import PosterDataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    class PosterDiffusionDataset(PosterDataset):
        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), self.labels[idx]

    ds = PosterDiffusionDataset(data_dir, split="train")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

@torch.inference_mode()
def log_samples(
    model: UNet,
    scheduler: DiffusionScheduler,
    device: torch.device,
    n_samples: int = 16,
    image_size: int = 32,
    epoch: int = 0,
    n_classes: int = 10,
    use_ddim: bool = True,
):
    """Generate and log a grid of samples to MLflow."""
    model.eval()
    class_labels = torch.arange(min(n_samples, n_classes), device=device)
    if len(class_labels) < n_samples:
        class_labels = class_labels.repeat(n_samples // len(class_labels) + 1)[:n_samples]

    shape = (n_samples, 3, image_size, image_size)
    if use_ddim:
        samples = ddim_sample(model, scheduler, shape, device, class_labels, n_steps=50)
    else:
        samples = ddpm_sample(model, scheduler, shape, device, class_labels)

    samples = (samples.clamp(-1, 1) + 1) / 2  # → [0, 1]
    grid = make_grid(samples, nrow=4)
    out_path = SAMPLE_DIR / f"epoch_{epoch:04d}.png"
    save_image(grid, out_path)
    mlflow.log_artifact(str(out_path), artifact_path="samples")
    model.train()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    dataset: str = "mnist",
    data_dir: Path = Path("data/raw/tmdb/posters"),
    epochs: int = 100,
    batch_size: int = 32,
    image_size: int = 32,
    lr: float = 2e-4,
    T: int = 1000,
    schedule: str = "cosine",
    n_classes: int | None = 10,
    sample_every: int = 10,
    grad_clip: float = 1.0,
    sample_size: int | None = None,   # limit dataset size for CPU runs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    # Data
    if dataset == "mnist":
        loader = get_mnist_loader(image_size, batch_size, sample_size)
        n_classes = 10
    else:
        loader = get_tmdb_loader(data_dir, image_size, batch_size)
        n_classes = 8  # 8 TMDB genres

    # Model + scheduler
    # Smaller model for CPU runs — reduces from 25M to ~3M parameters
    model = UNet(
        in_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 4),
        n_res_blocks=2,
        n_classes=n_classes,
        time_emb_dim=128,
    ).to(device)

    scheduler = DiffusionScheduler(T=T, schedule=schedule).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    mlflow.set_experiment("cineml-diffusion")
    with mlflow.start_run():
        mlflow.log_params({
            "dataset": dataset, "epochs": epochs, "batch_size": batch_size,
            "image_size": image_size, "lr": lr, "T": T, "schedule": schedule,
            "n_params": n_params, "n_classes": n_classes,
        })

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                x0, labels = batch
                x0 = x0.to(device)
                labels = labels.to(device) if n_classes else None

                loss = compute_loss(model, scheduler, x0, labels)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()
                total_loss += loss.item()

            lr_scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f"PROGRESS:{epoch}/{epochs}:Training DDPM  loss={avg_loss:.5f}", flush=True)
            log.info("Epoch %d  loss=%.5f  lr=%.2e", epoch, avg_loss,
                     lr_scheduler.get_last_lr()[0])
            mlflow.log_metrics({"loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=epoch)

            # Periodic sampling
            if epoch % sample_every == 0 or epoch == 1:
                log_samples(model, scheduler, device, n_classes=n_classes or 10,
                            image_size=image_size, epoch=epoch)

            # Checkpoint
            ckpt = MODEL_DIR / f"ddpm_{dataset}_ep{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "loss": avg_loss,
                "config": {
                    "base_channels": 32,
                    "channel_mults": (1, 2, 4),
                    "n_res_blocks": n_res,
                    "time_emb_dim": 128,
                    "n_classes": n_classes,
                },
            }, ckpt)
            mlflow.log_artifact(str(ckpt), artifact_path="checkpoints")

    log.info("Training complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on MNIST or TMDB posters")
    parser.add_argument("--dataset", choices=["mnist", "tmdb"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/tmdb/posters"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit dataset to N samples (e.g. 5000). None = full dataset.")
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        T=args.T,
        schedule=args.schedule,
        sample_every=args.sample_every,
        sample_size=args.sample_size,
    )
