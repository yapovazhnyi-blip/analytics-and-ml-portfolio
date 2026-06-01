"""
Module 5 — U-Net architecture for DDPM.

A class-conditional U-Net with:
  - Sinusoidal timestep embeddings
  - ResNet blocks with GroupNorm + FiLM conditioning
  - Cross-attention at the bottleneck for genre conditioning
  - Explicit skip-connection channel tracking (fixes channel mismatch)

Reference: Ho et al. (2020) — Denoising Diffusion Probabilistic Models
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ─────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.d_model // 2
        freqs  = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """ResNet block with FiLM time-step conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        # Use min(groups, in_ch) so GroupNorm works even for small channel counts
        g_in  = min(groups, in_ch)
        g_out = min(groups, out_ch)
        self.norm1   = nn.GroupNorm(g_in,  in_ch)
        self.conv1   = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2   = nn.GroupNorm(g_out, out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    """Spatial features (Q) attend to genre conditioning tokens (K, V)."""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.scale   = (channels // n_heads) ** -0.5
        g = min(8, channels)
        self.norm  = nn.GroupNorm(g, channels)
        self.to_q  = nn.Linear(channels, channels)
        self.to_k  = nn.Linear(context_dim, channels)
        self.to_v  = nn.Linear(context_dim, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)   # (B, HW, C)
        Q = self.to_q(h)
        K = self.to_k(context)
        V = self.to_v(context)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.n_heads, C // self.n_heads).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, H * W, C)
        return x + self.to_out(out).transpose(1, 2).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── U-Net ──────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Class-conditional U-Net for DDPM.

    Channel tracking strategy (fixes the GroupNorm mismatch):
      During __init__ we walk the encoder and record the output channel count
      of every ResBlock into `skip_channels`. During the decoder we pop from
      that list to know exactly how many channels each skip connection adds,
      and initialise each decoder ResBlock with `ch + skip_ch` input channels.
      This guarantees GroupNorm sees the correct channel count at every layer.

    Architecture (default, base_channels=64, channel_mults=(1,2,4,8)):
      Input conv : C_in -> 64
      Encoder    : 64 -> 128 -> 256 -> 512   (ResBlock x2 per level + Downsample)
      Bottleneck : 512 -> CrossAttn -> 512
      Decoder    : 512+skip -> 256 -> 128 -> 64  (Upsample + ResBlock x2 per level)
      Output conv: 64 -> C_in
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        base_channels: int   = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        n_res_blocks:  int   = 2,
        n_classes:     int | None = None,
        class_emb_dim: int   = 128,
        time_emb_dim:  int   = 256,
    ) -> None:
        super().__init__()

        # ── Time embedding ─────────────────────────────────────────────────────
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        cond_dim = time_emb_dim

        # ── Class conditioning ─────────────────────────────────────────────────
        self.class_emb  = None
        self.class_proj = None
        self._class_emb_dim = class_emb_dim
        if n_classes is not None:
            self.class_emb  = nn.Embedding(n_classes, class_emb_dim)
            self.class_proj = nn.Linear(class_emb_dim, time_emb_dim)

        # ── Input projection ───────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        ch = base_channels

        # ── Encoder ────────────────────────────────────────────────────────────
        # We record every ResBlock's output channel count so the decoder knows
        # exactly what to expect from each skip connection.
        self.encoder_blocks = nn.ModuleList()
        self.downsamples    = nn.ModuleList()
        skip_channels: list[int] = []   # channel count of each skip connection

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(n_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, cond_dim))
                skip_channels.append(out_ch)
                ch = out_ch
            # Downsample between every level except the last
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))

        # ── Bottleneck ─────────────────────────────────────────────────────────
        btn_context_dim = class_emb_dim if n_classes is not None else cond_dim
        self.mid_block1 = ResBlock(ch, ch, cond_dim)
        self.mid_attn   = CrossAttention(ch, btn_context_dim)
        self.mid_block2 = ResBlock(ch, ch, cond_dim)

        # ── Decoder ────────────────────────────────────────────────────────────
        # Mirror the encoder in reverse. For each level:
        #   1. Upsample (except when coming from the very last encoder level)
        #   2. For each ResBlock, concatenate the matching skip, then apply ResBlock.
        #      The first ResBlock at each level takes (ch + skip_ch) input channels;
        #      subsequent ResBlocks at the same level take (out_ch) input channels.
        self.upsamples      = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            # Upsample before every decoder level except the first
            # (the first decoder level corresponds to the deepest encoder level,
            #  which had no downsample)
            if i > 0:
                self.upsamples.append(Upsample(ch))

            for j in range(n_res_blocks):
                skip_ch = skip_channels.pop()
                in_ch   = ch + skip_ch   # skip concatenated on every ResBlock
                self.decoder_blocks.append(ResBlock(in_ch, out_ch, cond_dim))
                ch = out_ch

        # ── Output ─────────────────────────────────────────────────────────────
        g = min(8, ch)
        self.output_norm = nn.GroupNorm(g, ch)
        self.output_conv = nn.Conv2d(ch, in_channels, 1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:            torch.Tensor,
        t:            torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x            : noisy image  (B, C, H, W)
            t            : timesteps    (B,)
            class_labels : genre IDs    (B,)  — optional
        Returns: predicted noise ε̂     (B, C, H, W)
        """
        # ── Conditioning ───────────────────────────────────────────────────────
        t_emb   = self.time_emb(t)                        # (B, time_dim)
        context = None
        if self.class_emb is not None and class_labels is not None:
            cls_emb  = self.class_emb(class_labels)       # (B, class_emb_dim)
            t_emb    = t_emb + self.class_proj(cls_emb)   # add to time emb
            context  = cls_emb.unsqueeze(1)               # (B, 1, class_emb_dim)
        elif self.mid_attn is not None:
            # No class label: use zero context so cross-attention is a no-op
            B = x.shape[0]
            context = torch.zeros(B, 1, self._class_emb_dim, device=x.device)

        # ── Encode ─────────────────────────────────────────────────────────────
        h      = self.input_conv(x)
        skips  = [h]                   # save every feature map as a skip
        enc_it = iter(self.encoder_blocks)
        ds_it  = iter(self.downsamples)

        n_levels  = len(self.downsamples) + 1   # total encoder levels
        n_res     = len(self.encoder_blocks) // n_levels  # res blocks per level

        for level in range(n_levels):
            for _ in range(n_res):
                h = next(enc_it)(h, t_emb)
                skips.append(h)
            if level < n_levels - 1:
                h = next(ds_it)(h)

        # ── Bottleneck ─────────────────────────────────────────────────────────
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        # ── Decode ─────────────────────────────────────────────────────────────
        dec_it = iter(self.decoder_blocks)
        up_it  = iter(self.upsamples)

        for i in range(n_levels):
            if i > 0:
                h = next(up_it)(h)
            for _ in range(n_res):
                skip = skips.pop()
                h    = torch.cat([h, skip], dim=1)
                h    = next(dec_it)(h, t_emb)

        return self.output_conv(F.silu(self.output_norm(h)))
