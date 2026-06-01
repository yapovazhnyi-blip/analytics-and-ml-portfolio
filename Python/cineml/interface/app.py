"""
CineML -- Unified Interface
Run: streamlit run interface/app.py
"""
import sys
from pathlib import Path

_HERE       = Path(__file__).resolve().parent        # interface/
ROOT        = _HERE.parent                           # cineml/
_COMPONENTS = _HERE / "components"

for _p in [str(ROOT), str(_HERE), str(_COMPONENTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="CineML Platform",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "🏠  Overview":        "pages/01_overview.py",
    "🎯  Recommender":     "pages/02_recommender.py",
    "📊  A/B Engine":      "pages/03_ab_engine.py",
    "🔬  Analysis Memo":   "pages/04_analysis.py",
    "🎨  Diffusion + ViT": "pages/05_diffusion.py",
}

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0 0 16px'>"
        "<span style='font-size:32px'>🎬</span><br>"
        "<strong style='font-size:16px'>CineML</strong><br>"
        "<span style='font-size:12px;opacity:.6'>Streaming Content Intelligence</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Module status")
    _checks = {
        "M1 data":      (ROOT / "data/processed/ratings.parquet").exists(),
        "M1 events":    (ROOT / "data/processed/streaming_events.parquet").exists(),
        "M2 ALS":       (ROOT / "data/models/als_model.pkl").exists(),
        "M2 two-tower": (ROOT / "data/models/two_tower.pt").exists(),
        "M5 ViT":       (ROOT / "data/models/vit_genre").exists(),
        "M5 DDPM":      bool(list((ROOT / "data/models").glob("ddpm_*.pt")))
                        if (ROOT / "data/models").exists() else False,
    }
    for _label, _ok in _checks.items():
        st.caption(f"{'🟢' if _ok else '🔴'}  {_label}")

page_path = _HERE / PAGES[page]
_ns = {"__file__": str(page_path), "__name__": "__main__", "__builtins__": __builtins__}
with open(page_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(page_path), "exec"), _ns)
