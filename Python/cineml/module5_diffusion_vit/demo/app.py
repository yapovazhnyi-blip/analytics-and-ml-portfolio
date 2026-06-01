"""
Module 5 — Gradio Demo: Generate movie poster art from genre.

Runs both:
  1. ViT genre classifier — upload a poster, get genre predictions + attention rollout
  2. DDPM/DDIM generator — select genre, generate synthetic poster artwork

Usage:
    python module5_diffusion_vit/demo/app.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent.parent))

from module5_diffusion_vit.diffusion.unet import UNet
from module5_diffusion_vit.diffusion.ddpm import DiffusionScheduler, ddim_sample

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Horror", "Documentary"]
IMAGE_SIZE = 64
MODEL_DIR = Path("data/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_diffusion_model() -> tuple[UNet, DiffusionScheduler] | tuple[None, None]:
    ckpt_paths = sorted(MODEL_DIR.glob("ddpm_tmdb_*.pt"))
    if not ckpt_paths:
        ckpt_paths = sorted(MODEL_DIR.glob("ddpm_mnist_*.pt"))
    if not ckpt_paths:
        return None, None

    ckpt = torch.load(ckpt_paths[-1], map_location=DEVICE)
    sd = ckpt.get("model_state_dict", ckpt)

    # Load config saved by train.py if available (most reliable)
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None

    if cfg:
        base_ch  = cfg["base_channels"]
        time_dim = cfg["time_emb_dim"]
        n_cls    = cfg.get("n_classes")
        mults    = tuple(cfg["channel_mults"])
        n_res    = cfg["n_res_blocks"]
    else:
        # Infer from checkpoint tensor shapes
        base_ch  = sd["input_conv.weight"].shape[0]
        time_dim = sd["time_emb.1.bias"].shape[0]
        n_cls    = sd["class_emb.weight"].shape[0] if "class_emb.weight" in sd else None
        # Count unique module indices (each module has weight + bias keys)
        n_ds     = len(set(k.split(".")[1] for k in sd if k.startswith("downsamples.")))
        n_levels = n_ds + 1
        n_dec    = len(set(k.split(".")[1] for k in sd if k.startswith("decoder_blocks.")))
        n_res    = n_dec // n_levels
        # Encoder output channels — last block of each level
        enc_out_chs = [sd[k].shape[0] for k in sorted(sd)
                       if k.startswith("encoder_blocks.") and k.endswith(".conv2.weight")]
        mults = tuple(enc_out_chs[i] // base_ch for i in range(n_res - 1, len(enc_out_chs), n_res))

    import logging as _log
    _log.getLogger(__name__).warning(
        "DIFFUSION CONFIG: base_ch=%d time_dim=%d n_cls=%s mults=%s n_res=%d",
        base_ch, time_dim, n_cls, mults, n_res
    )

    model = UNet(
        in_channels=3, base_channels=base_ch,
        channel_mults=mults, n_res_blocks=n_res,
        n_classes=n_cls, time_emb_dim=time_dim,
    ).to(DEVICE)

    model.load_state_dict(sd)
    model.eval()
    scheduler = DiffusionScheduler(T=1000, schedule="cosine").to(DEVICE)
    return model, scheduler


def load_vit_model():
    """Load fine-tuned ViT or fall back to a pretrained baseline."""
    try:
        from transformers import ViTForImageClassification
        vit_path = MODEL_DIR / "vit_genre"
        if vit_path.exists():
            model = ViTForImageClassification.from_pretrained(str(vit_path))
        else:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=len(GENRES),
                ignore_mismatched_sizes=True,
            )
        model.eval().to(DEVICE)
        return model
    except Exception:
        return None


# ── Inference functions ────────────────────────────────────────────────────────

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def classify_poster(image: Image.Image):
    """Run ViT genre classification + attention rollout."""
    vit = load_vit_model()
    if vit is None:
        return None, "ViT model not available. Run module5_diffusion_vit/vit/finetune.py first."

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = vit(pixel_values=tensor)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Attention rollout
    from module5_diffusion_vit.vit.finetune import AttentionRollout
    rollout = AttentionRollout(vit)
    try:
        attn_map = rollout(tensor)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image.resize((224, 224)))
        axes[0].set_title("Input Poster")
        axes[0].axis("off")
        im = axes[1].imshow(attn_map, cmap="inferno", interpolation="bicubic")
        axes[1].set_title("Attention Rollout — which patches matter")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        fig.canvas.draw()
        rollout_img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
    except Exception as e:
        rollout_img = None

    # Bar chart of genre probabilities
    genre_probs = {g: float(p) for g, p in zip(GENRES, probs)}
    return rollout_img, genre_probs


def generate_poster(genre: str, n_steps: int, eta: float, seed: int):
    """Generate a synthetic poster for the selected genre using DDIM."""
    diffusion_model, scheduler = load_diffusion_model()
    if diffusion_model is None:
        placeholder = Image.fromarray(
            np.random.randint(50, 200, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        )
        return placeholder, "⚠️ No trained model found. Showing random noise. Train with diffusion/train.py"

    torch.manual_seed(seed)
    class_idx = GENRES.index(genre)
    label = torch.tensor([class_idx], device=DEVICE)
    shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        sample = ddim_sample(
            diffusion_model, scheduler, shape, DEVICE,
            class_labels=label, n_steps=n_steps, eta=eta,
        )

    img_array = ((sample[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    img = Image.fromarray(img_array).resize((256, 256), Image.NEAREST)
    return img, f"Generated {genre} poster ({n_steps} DDIM steps, η={eta})"


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="CineML — Diffusion + ViT Demo") as demo:
    gr.Markdown("""
    # 🎬 CineML — Generative Artwork Module
    **Module 5**: Vision Transformer genre classifier + DDPM/DDIM poster generator

    *Part of the [CineML Platform](https://github.com/yourname/cineml) — End-to-End Streaming Content Intelligence*
    """)

    with gr.Tabs():
        # ── Tab 1: Generator ──────────────────────────────────────────────────
        with gr.TabItem("🎨 Generate Poster"):
            gr.Markdown("### Generate synthetic movie poster artwork by genre")
            with gr.Row():
                with gr.Column():
                    genre_dropdown = gr.Dropdown(GENRES, value="Sci-Fi", label="Genre")
                    n_steps_slider = gr.Slider(10, 200, value=50, step=10, label="DDIM steps")
                    eta_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="η (0=deterministic, 1=DDPM)")
                    seed_number = gr.Number(value=42, label="Random seed", precision=0)
                    generate_btn = gr.Button("✨ Generate", variant="primary")
                with gr.Column():
                    gen_output = gr.Image(label="Generated Poster", type="pil")
                    gen_caption = gr.Textbox(label="Info")

            generate_btn.click(
                generate_poster,
                inputs=[genre_dropdown, n_steps_slider, eta_slider, seed_number],
                outputs=[gen_output, gen_caption],
            )

        # ── Tab 2: Classifier ─────────────────────────────────────────────────
        with gr.TabItem("🔍 Classify & Explain"):
            gr.Markdown("### Upload a movie poster — get genre predictions + attention rollout")
            with gr.Row():
                with gr.Column():
                    poster_input = gr.Image(label="Upload Poster", type="pil")
                    classify_btn = gr.Button("🔬 Classify", variant="primary")
                with gr.Column():
                    rollout_output = gr.Image(label="Attention Rollout")
                    genre_bar = gr.Label(label="Genre Probabilities", num_top_classes=5)

            classify_btn.click(
                classify_poster,
                inputs=[poster_input],
                outputs=[rollout_output, genre_bar],
            )

    gr.Markdown("""
    ---
    **Architecture notes:**
    - Generator: Class-conditional U-Net (Ho et al. 2020) with cross-attention genre conditioning
    - Classifier: Fine-tuned `google/vit-base-patch16-224` (Dosovitskiy et al. 2021)
    - Attention Rollout: Abnar & Zuidema (2020) — patch-level saliency via attention matrix product
    """)


if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0", share=False, theme=gr.themes.Soft())
