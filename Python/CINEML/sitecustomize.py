"""
sitecustomize.py
================
Python loads this file automatically at interpreter startup if it exists
on sys.path. Placing it in the repo root means any script run from this
directory (or with PYTHONPATH pointing here) gets UTF-8 stdout/stderr
automatically -- fixing cp1251/cp1252 errors on Windows without changing
individual scripts.

This is the standard pattern for per-project encoding fixes on Windows.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
