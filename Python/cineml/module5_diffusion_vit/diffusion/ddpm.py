"""
Module 5 — DDPM & DDIM Samplers (from scratch).

Implements the full forward/reverse diffusion process following:
  Ho et al. (2020) — Denoising Diffusion Probabilistic Models
  Song et al. (2021) — Denoising Diffusion Implicit Models (DDIM)

Key equations:
  Forward:  q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ)xₜ₋₁, βₜI)
  Reparameterised: q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1−ᾱₜ)I)
  Reverse (learned): pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ̃θ(xₜ, t), σ²ₜI)
  Loss: Lsimple = E[‖ε − εθ(xₜ, t)‖²]
"""
import math
from typing import Callable

import torch
import torch.nn.functional as F


class DiffusionScheduler:
    """
    Pre-computes all noise schedule constants for T timesteps.
    Supports linear and cosine schedules.
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.T = T

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos((steps / T + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], max=0.999)
            betas = betas.float()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)           # ᾱₜ
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.log_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).log()
        self.sqrt_recip_alphas = (1.0 / self.alphas).sqrt()

        # Posterior variance q(xₜ₋₁ | xₜ, x₀)
        self.posterior_variance = (
            betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * self.alphas_cumprod_prev.sqrt() / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionScheduler":
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self


# ── Training utilities ────────────────────────────────────────────────────────

def q_sample(
    scheduler: DiffusionScheduler,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion: sample xₜ given x₀ and t.
    Returns (xₜ, noise_applied)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ac = scheduler.sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_ac * x0 + sqrt_one_minus * noise, noise


def compute_loss(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    x0: torch.Tensor,
    class_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simple MSE loss on noise prediction: E[‖ε − εθ(xₜ, t)‖²]
    """
    B = x0.shape[0]
    device = x0.device
    t = torch.randint(0, scheduler.T, (B,), device=device)
    noise = torch.randn_like(x0)
    xt, _ = q_sample(scheduler, x0, t, noise)
    predicted_noise = model(xt, t, class_labels)
    return F.mse_loss(predicted_noise, noise)


# ── DDPM sampler ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def ddpm_sample(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    shape: tuple[int, ...],
    device: torch.device,
    class_labels: torch.Tensor | None = None,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """
    Ancestral sampling: iterate xₜ → xₜ₋₁ for T steps.
    Returns final sample x₀ in range [-1, 1].
    """
    x = torch.randn(shape, device=device)

    for t_val in reversed(range(scheduler.T)):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
        predicted_noise = model(x, t, class_labels)

        # Compute x₀ prediction
        sqrt_recip = scheduler.sqrt_recip_alphas[t_val]
        sqrt_one_minus = scheduler.sqrt_one_minus_alphas_cumprod[t_val]
        beta_t = scheduler.betas[t_val]

        # Equation 11 from Ho et al.
        x_prev_mean = sqrt_recip * (x - beta_t / sqrt_one_minus * predicted_noise)

        if clip_denoised:
            x_prev_mean = x_prev_mean.clamp(-1.0, 1.0)

        if t_val > 0:
            noise = torch.randn_like(x)
            posterior_std = scheduler.posterior_variance[t_val].sqrt()
            x = x_prev_mean + posterior_std * noise
        else:
            x = x_prev_mean

    return x


# ── DDIM sampler ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def ddim_sample(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    shape: tuple[int, ...],
    device: torch.device,
    class_labels: torch.Tensor | None = None,
    n_steps: int = 50,
    eta: float = 0.0,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """
    DDIM deterministic sampler — much faster than DDPM (50 vs 1000 steps).

    η=0   → deterministic (original DDIM)
    η=1   → recovers DDPM stochasticity

    Reference: Song et al. (2021) Section 4.2
    """
    # Sub-sequence of timesteps
    step_ratio = scheduler.T // n_steps
    timesteps = list(reversed(range(0, scheduler.T, step_ratio)))

    x = torch.randn(shape, device=device)

    for i, t_val in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

        predicted_noise = model(x, t, class_labels)

        ac_t = scheduler.alphas_cumprod[t_val]
        ac_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x₀
        x0_pred = (x - (1 - ac_t).sqrt() * predicted_noise) / ac_t.sqrt()
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Direction towards xₜ
        sigma = eta * ((1 - ac_prev) / (1 - ac_t) * (1 - ac_t / ac_prev)).sqrt()
        direction = (1 - ac_prev - sigma ** 2).sqrt() * predicted_noise

        x = ac_prev.sqrt() * x0_pred + direction
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn_like(x)

    return x
