"""Streamlit Community Cloud entry point.

Streamlit Cloud reads requirements.txt from the entry point's folder, so this wrapper lets the
cloud app use CPU-only PyTorch (deploy/streamlit/requirements.txt) while the repo root keeps a
general requirements.txt for local use. It runs the real app from the repo root.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
