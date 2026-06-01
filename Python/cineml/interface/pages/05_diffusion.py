"""
Page 5 — Generative Artwork Module (M5)
ViT genre classifier + DDPM/DDIM poster generator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bootstrap import ROOT, COMPONENTS  # noqa: E402

import io
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms

from data_loader import load_diffusion, load_vit

ROOT = Path(__file__).parent.parent.parent

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi", "Romance", "Horror", "Documentary"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

st.title("Generative Artwork — Diffusion + ViT")
st.caption("Module 5 — DDPM from scratch · class-conditional U-Net · ViT fine-tuning · attention rollout")
st.divider()

tab_gen, tab_cls, tab_arch = st.tabs(
    ["🎨 Generate Poster", "🔍 Classify & Explain", "📐 Architecture"]
)


# ── Tab 1: Generate ────────────────────────────────────────────────────────────
with tab_gen:
    st.subheader("Generate synthetic movie poster art by genre")

    col_ctrl, col_out = st.columns([1, 1])
    with col_ctrl:
        genre    = st.selectbox("Genre", GENRES, index=4)
        n_steps  = st.slider("DDIM steps (fewer = faster)", 10, 200, 50, 10)
        eta      = st.slider("η — stochasticity (0 = deterministic DDIM, 1 = DDPM)",
                              0.0, 1.0, 0.0, 0.1)
        seed     = st.number_input("Seed", value=42, step=1)
        gen_btn  = st.button("✨ Generate", type="primary", use_container_width=True)

        st.caption("Sampler: η=0 → DDIM (Song et al. 2021), η=1 → DDPM (Ho et al. 2020)")

    with col_out:
        if gen_btn:
            import torch
            model, scheduler = load_diffusion()

            if model is None:
                st.warning(
                    "No trained DDPM checkpoint found. "
                    "Run the training script first:\n\n"
                    "```bash\n"
                    "# Quick CPU sanity check (~20 min)\n"
                    "python module5_diffusion_vit/diffusion/train.py \\\n"
                    "    --dataset mnist --epochs 20 --image-size 32\n"
                    "```"
                )
                # Show a placeholder noise sample
                rng   = np.random.default_rng(int(seed))
                noise = rng.integers(40, 200, (128, 128, 3), dtype=np.uint8)
                st.image(Image.fromarray(noise), caption="Random noise (no model loaded)",
                          use_column_width=True)
            else:
                from module5_diffusion_vit.diffusion.ddpm import ddim_sample, ddpm_sample
                device = torch.device("cpu")  # Streamlit runs on CPU
                torch.manual_seed(int(seed))

                genre_idx = GENRES.index(genre)
                label     = torch.tensor([genre_idx])
                # Infer image size from model
                img_size  = 64

                with st.spinner(f"Sampling ({n_steps} DDIM steps)…"):
                    shape = (1, 3, img_size, img_size)
                    sample = ddim_sample(
                        model, scheduler, shape, device,
                        class_labels=label, n_steps=n_steps, eta=eta,
                    )

                arr = ((sample[0].clamp(-1,1).permute(1,2,0).numpy() + 1) / 2 * 255).astype(np.uint8)
                img = Image.fromarray(arr).resize((256, 256), Image.NEAREST)
                st.image(img, caption=f"{genre} poster (seed={seed}, η={eta})")

                # Download button
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button("Download PNG", buf.getvalue(),
                                    file_name=f"cineml_{genre.lower()}_{seed}.png",
                                    mime="image/png")

    # Sample grid from training
    sample_dir = ROOT / "data" / "samples"
    if sample_dir.exists():
        samples = sorted(sample_dir.glob("epoch_*.png"))
        if samples:
            st.divider()
            st.subheader("Training progression")
            st.caption("Generated samples saved during training (latest checkpoints)")
            n_show = min(6, len(samples))
            selected = [samples[int(i * (len(samples)-1) / max(n_show-1,1))] for i in range(n_show)]
            img_cols = st.columns(n_show)
            for col, p in zip(img_cols, selected):
                epoch_n = p.stem.replace("epoch_", "")
                col.image(str(p), caption=f"Epoch {epoch_n}", use_column_width=True)


# ── Tab 2: Classify ────────────────────────────────────────────────────────────
with tab_cls:
    st.subheader("Genre classifier + attention rollout")
    st.caption(
        "Upload a movie poster. The fine-tuned ViT-B/16 predicts genre; "
        "Attention Rollout (Abnar & Zuidema 2020) shows which 16×16 patches drove the decision."
    )

    uploaded = st.file_uploader("Upload a movie poster (JPG / PNG)", type=["jpg", "jpeg", "png"])
    cls_btn  = st.button("🔬 Classify", type="primary")

    if uploaded and cls_btn:
        import torch
        img = Image.open(uploaded).convert("RGB")
        vit = load_vit()

        col_img, col_rollout, col_probs = st.columns([1, 1, 1])
        with col_img:
            st.image(img.resize((224, 224)), caption="Input poster", use_column_width=True)

        if vit is None:
            st.info(
                "Fine-tune the ViT first:\n\n"
                "```bash\npython module5_diffusion_vit/vit/finetune.py "
                "--data-dir data/raw/tmdb/posters\n```"
            )
        else:
            tensor = PREPROCESS(img).unsqueeze(0)

            with torch.no_grad():
                outputs = vit(pixel_values=tensor)
                probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()

            # Attention rollout
            try:
                from module5_diffusion_vit.vit.finetune import AttentionRollout
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm

                rollout_fn = AttentionRollout(vit)
                attn_map   = rollout_fn(tensor)

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(attn_map, cmap="inferno", interpolation="bicubic")
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                with col_rollout:
                    st.image(buf, caption="Attention Rollout (14×14 patches)", use_column_width=True)
            except Exception as e:
                with col_rollout:
                    st.warning(f"Rollout failed: {e}")

            with col_probs:
                probs_df = {"Genre": GENRES, "Probability": probs.tolist()}
                import plotly.express as px
                fig = px.bar(probs_df, x="Probability", y="Genre",
                              orientation="h", height=260,
                              color="Probability",
                              color_continuous_scale="Purples")
                fig.update_layout(coloraxis_showscale=False,
                                   yaxis={"autorange": "reversed"},
                                   margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True)

                top_genre = GENRES[int(probs.argmax())]
                top_conf  = float(probs.max())
                st.metric("Predicted genre", top_genre, f"{top_conf:.1%} confidence")


# ── Tab 3: Architecture ────────────────────────────────────────────────────────
with tab_arch:
    st.subheader("Architecture notes")

    with st.expander("U-Net & noise schedule", expanded=True):
        st.markdown("""
**Forward diffusion** (Ho et al. 2020):
```
q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1−ᾱₜ)I)
```
A cosine noise schedule (Nichol & Dhariwal 2021) linearly increases noise while preserving fine detail longer than the original linear schedule.

**U-Net** predicts the noise `ε` at each timestep:
```
Input: xₜ (noisy image)  +  t (timestep)  +  c (genre label)
 ↓
Sinusoidal time embedding  →  FiLM conditioning into every ResBlock
Genre embedding            →  Cross-attention at bottleneck
 ↓
Enc: 64→128→256→512 channels   (ResBlock × 2 + Downsample)
Bottleneck: ResBlock + Cross-Attention + ResBlock
Dec: 512→256→128→64 channels   (ResBlock × 2 + Upsample + skip)
 ↓
Output: ε̂ (B, C, H, W)
```

**Cross-attention conditioning** — genre embedding → K, V; spatial features → Q. This is the same mechanism used in Stable Diffusion and Sora.
""")

    with st.expander("DDIM sampler — why 50 steps instead of 1000"):
        st.markdown("""
DDIM (Song et al. 2021) rewrites the reverse process as a **non-Markovian chain**. Because the denoising trajectory is no longer tied to individual Markov steps, you can skip most of the 1,000 timesteps and still reach the same result.

```python
# DDIM reverse step
x0_pred = (x - √(1−ᾱₜ) · ε̂) / √ᾱₜ
direction = √(1−ᾱₜ₋₁ − σ²) · ε̂
x_prev = √ᾱₜ₋₁ · x0_pred + direction + σ · noise
```

Setting `η = 0` → `σ = 0` → fully deterministic. Same seed always gives same image.
Setting `η = 1` → recovers DDPM stochasticity.
""")

    with st.expander("ViT & Attention Rollout"):
        st.markdown("""
The Vision Transformer splits a 224×224 image into **196 non-overlapping 16×16 patches**, each projected to a 768-dim vector (for ViT-B). These patches — plus a learnable `[CLS]` token — are fed through 12 Transformer encoder layers.

**Attention Rollout** (Abnar & Zuidema 2020):

1. Extract the 12×12 attention matrices (one per head) from every layer.
2. Average across heads → one 197×197 matrix per layer.
3. Add the residual connection: `A' = (A + I) / 2` (accounts for the skip path).
4. Multiply across all layers: `R = A'_L · A'_{L-1} · … · A'_1`.
5. The `[CLS]` row of `R` gives the weight of each patch to the final prediction.
6. Reshape to 14×14 → normalise → overlay on original image.

The result shows *which patches the model used to make its decision* — a lightweight explainability method requiring no gradient computation.
""")

    # Key references
    st.divider()
    st.subheader("Key references")
    refs = [
        ("Ho et al. 2020", "Denoising Diffusion Probabilistic Models",
         "DDPM — forward/reverse process, simplified noise loss"),
        ("Song et al. 2021", "Denoising Diffusion Implicit Models",
         "DDIM — deterministic sampler, large step skipping"),
        ("Nichol & Dhariwal 2021", "Improved Denoising Diffusion Probabilistic Models",
         "Cosine noise schedule, learned variance"),
        ("Dosovitskiy et al. 2021", "An Image is Worth 16×16 Words",
         "ViT — patch embeddings, multi-head self-attention"),
        ("Abnar & Zuidema 2020", "Quantifying Attention Flow in Transformers",
         "Attention Rollout — patch saliency without gradients"),
    ]
    for authors, title, contribution in refs:
        st.markdown(f"**{authors}** — *{title}*  \n{contribution}")
