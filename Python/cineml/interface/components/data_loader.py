"""
interface/components/data_loader.py
====================================
Shared @st.cache_resource loaders for models and data.
All pages import from here so models are loaded once per session.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "data" / "processed"
MODEL_DIR  = ROOT / "data" / "models"
SAMPLE_DIR = ROOT / "data" / "samples"


# ── Data ───────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_ratings() -> pd.DataFrame | None:
    p = DATA_DIR / "ratings.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame | None:
    p = DATA_DIR / "movies.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_events(sample: int = 50_000) -> pd.DataFrame:
    p = DATA_DIR / "streaming_events.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df.sample(min(sample, len(df)), random_state=42)
    # ── Fallback: synthetic demo data ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    n = 20_000
    users = np.arange(1000)
    movies = np.arange(500)
    records = []
    for _ in range(n):
        uid  = rng.choice(users)
        mid  = rng.choice(movies)
        arm  = "treatment" if uid % 2 == 0 else "control"
        ctr  = rng.beta(3, 17) * (1.15 if arm == "treatment" else 1.0)
        etype = "impression"
        records.append({"user_id": uid, "movie_id": mid, "event_type": etype,
                         "arm": arm, "dwell_ms": int(rng.lognormal(7, 1)),
                         "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))})
        if rng.random() < ctr:
            records.append({**records[-1], "event_type": "click",
                             "dwell_ms": int(rng.lognormal(8.5, 1.2))})
            if rng.random() < 0.6:
                records.append({**records[-1], "event_type": "completion",
                                 "dwell_ms": int(rng.lognormal(12, 0.5))})
            elif rng.random() < 0.5:
                records.append({**records[-1], "event_type": "skip",
                                 "dwell_ms": int(rng.lognormal(6, 1))})
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_tmdb() -> pd.DataFrame | None:
    p = ROOT / "data" / "raw" / "tmdb" / "tmdb_metadata.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ── Models ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_als():
    import sys
    p = MODEL_DIR / "als_model.pkl"
    if not p.exists():
        return None
    # The pickle stores class references to 'models.als_model'.
    # Adding module2_recommender/ to sys.path lets pickle resolve that path.
    m2_path = str(ROOT / "module2_recommender")
    if m2_path not in sys.path:
        sys.path.insert(0, m2_path)
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_two_tower():
    # Check file exists before importing torch — avoids DLL errors on Windows
    # when torch isn't properly installed or model hasn't been trained yet.
    p = MODEL_DIR / "two_tower.pt"
    if not p.exists():
        return None

    try:
        import torch
        from module2_recommender.models.two_tower import TwoTowerModel
    except (ImportError, OSError):
        return None

    cfg_p = MODEL_DIR / "two_tower_config.json"
    if not p.exists():
        return None

    # Load config from file if available
    if cfg_p.exists():
        import json
        cfg = json.load(open(cfg_p, encoding="utf-8"))
    else:
        # Infer dims directly from checkpoint to avoid size mismatch
        import json
        ckpt = torch.load(p, map_location="cpu")
        cfg = {
            "n_users":  ckpt["user_tower.user_embed.weight"].shape[0],
            "n_items":  ckpt["item_tower.item_embed.weight"].shape[0],
            "n_genres": ckpt["item_tower.genre_embed.weight"].shape[0],
            "d_embed":  ckpt["user_tower.user_embed.weight"].shape[1],
            "d_model":  ckpt["user_tower.net.0.weight"].shape[0],
        }

    model = TwoTowerModel(**cfg)
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_vit():
    try:
        from transformers import ViTForImageClassification
        p = MODEL_DIR / "vit_genre"
        if p.exists():
            return ViTForImageClassification.from_pretrained(str(p)).eval()
        return ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=8,
            ignore_mismatched_sizes=True,
        ).eval()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_diffusion():
    try:
        import torch
        from module5_diffusion_vit.diffusion.unet import UNet
        from module5_diffusion_vit.diffusion.ddpm import DiffusionScheduler
    except (ImportError, OSError):
        return None, None

    ckpts = sorted(MODEL_DIR.glob("ddpm_tmdb_*.pt")) or sorted(MODEL_DIR.glob("ddpm_mnist_*.pt"))
    if not ckpts:
        return None, None

    state = torch.load(ckpts[-1], map_location="cpu")
    sd = state.get("model_state_dict", state)

    # Infer architecture from checkpoint tensor shapes
    # input_conv.weight shape: (base_channels, in_channels, 3, 3)
    base_ch = sd["input_conv.weight"].shape[0]
    # time_emb.1.weight shape: (time_emb_dim, time_emb_dim//4)
    time_dim = sd["time_emb.1.bias"].shape[0]
    # class_emb.weight shape: (n_classes, class_emb_dim) — may not exist
    n_classes = sd["class_emb.weight"].shape[0] if "class_emb.weight" in sd else None
    # Count unique module indices (weight + bias = 2 keys per module — don't double-count)
    n_ds     = len(set(k.split(".")[1] for k in sd if k.startswith("downsamples.")))
    n_levels = n_ds + 1
    n_dec    = len(set(k.split(".")[1] for k in sd if k.startswith("decoder_blocks.")))
    n_res    = n_dec // n_levels
    enc_out_chs = [sd[k].shape[0] for k in sorted(sd)
                   if k.startswith("encoder_blocks.") and k.endswith(".conv2.weight")]
    mults = tuple(enc_out_chs[i] // base_ch for i in range(n_res - 1, len(enc_out_chs), n_res))

    model = UNet(
        in_channels=3,
        base_channels=base_ch,
        channel_mults=tuple(mults),
        n_res_blocks=n_res,
        n_classes=n_classes,
        time_emb_dim=time_dim,
    )
    model.load_state_dict(sd)
    model.eval()
    scheduler = DiffusionScheduler(T=1000, schedule="cosine")
    return model, scheduler


# ── Helpers ────────────────────────────────────────────────────────────────────

def data_status() -> dict[str, bool]:
    return {
        "ratings":  (DATA_DIR / "ratings.parquet").exists(),
        "movies":   (DATA_DIR / "movies.parquet").exists(),
        "events":   (DATA_DIR / "streaming_events.parquet").exists(),
        "als":      (MODEL_DIR / "als_model.pkl").exists(),
        "two_tower":(MODEL_DIR / "two_tower.pt").exists(),
        "vit":      (MODEL_DIR / "vit_genre").exists(),
        "ddpm":     bool(list(MODEL_DIR.glob("ddpm_*.pt"))) if MODEL_DIR.exists() else False,
    }
