# Module 5 — Generative Artwork (Diffusion + ViT)

Given a movie genre, generate promotional poster artwork. Two components: a DDPM diffusion model implemented from scratch and a fine-tuned ViT genre classifier with attention rollout explainability.

## Training results

| Model | Config | Result |
|-------|--------|--------|
| DDPM MNIST | base_channels=16, channel_mults=(1,2,4), 20 epochs, 5k samples | loss=0.149 ✅ |
| ViT-B/16 | 8 genre classes, fine-tuned on TMDB posters | in progress |

## Run

```bash
# DDPM on MNIST (CPU sanity check, ~20 min)
python module5_diffusion_vit/diffusion/train.py \
    --dataset mnist --epochs 20 --batch-size 64 \
    --image-size 32 --sample-size 5000

# Sort TMDB posters by genre (one-time)
python module5_diffusion_vit/vit/sort_posters_by_genre.py

# Fine-tune ViT (needs sorted posters)
python module5_diffusion_vit/vit/finetune.py \
    --data-dir data/raw/tmdb/posters_by_genre \
    --epochs 5 --sample-size 500

# Gradio demo
python module5_diffusion_vit/demo/app.py
```

## Architecture: DDPM U-Net

```
Input: noisy image xₜ + timestep t + genre label c
  │
  ├── Sinusoidal time embedding → FiLM conditioning into every ResBlock
  └── Genre embedding → Cross-attention at bottleneck ← same as Stable Diffusion
  │
  Encoder: base_ch → 2×base_ch → 4×base_ch   (ResBlock + Downsample)
  Bottleneck: ResBlock + CrossAttention + ResBlock
  Decoder: 4×base_ch → 2×base_ch → base_ch   (ResBlock + Upsample + skip)
  │
  Output: predicted noise ε̂
```

**Channel tracking**: A `skip_channels` list built during `__init__` records every encoder ResBlock's output channel count, so decoder ResBlocks are initialised with the exact correct `in_ch` after skip concatenation. This prevents the GroupNorm size mismatch that plagued the original implementation.

## Samplers

| Sampler | Steps | Speed | Deterministic |
|---------|-------|-------|---------------|
| DDPM | 1000 | slow | no |
| DDIM (η=0) | 50 | 20× faster | yes |
| DDIM (η=1) | 50 | 20× faster | no (recovers DDPM) |

## Attention Rollout

Produces a 14×14 patch saliency map showing which image regions the ViT used to classify genre. Computed by multiplying attention matrices across all 12 Transformer layers with residual connection accounting — no gradient computation required.

## Key references

- Ho et al. (2020) — Denoising Diffusion Probabilistic Models
- Song et al. (2021) — Denoising Diffusion Implicit Models
- Nichol & Dhariwal (2021) — Improved DDPMs (cosine schedule)
- Dosovitskiy et al. (2021) — An Image is Worth 16×16 Words
- Abnar & Zuidema (2020) — Quantifying Attention Flow in Transformers
