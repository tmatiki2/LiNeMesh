"""
Download the pretrained LiNeMesh weights to weights/mod_brdgnet_ptrain.pth.

    python scripts/download_weights.py

(01_LM_mesh_viz.py also downloads them automatically on first run.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LM_paths import ensure_weights

if __name__ == "__main__":
    print("weights ready:", ensure_weights())
