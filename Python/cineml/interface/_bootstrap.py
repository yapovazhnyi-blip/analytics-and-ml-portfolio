"""
interface/_bootstrap.py  -- import path helper for all pages.
Import this as the first thing in every page file.
"""
import sys
from pathlib import Path

_INTERFACE  = Path(__file__).resolve().parent   # interface/
ROOT        = _INTERFACE.parent                 # cineml/
COMPONENTS  = _INTERFACE / "components"

for _p in [str(ROOT), str(_INTERFACE), str(COMPONENTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
