"""
Central path and download configuration for LiNeMesh.

Every script and the training notebook import their paths from here, so the
repository works out of the box once it is cloned: nothing has to be edited.

Default layout (all relative to the cloned repository):

    LiNeMesh/
    ├── data/bridgenet/train/<id>/<id>_render/   BridgeNet training domains
    ├── data/bridgenet/test/<id>/<id>_render/    BridgeNet test domains
    ├── data/synthetic/train/<id>/<id>_render/   domains made by 01_LM_synthetic_data.py
    ├── weights/mod_brdgnet_ptrain.pth           pretrained LiNeMesh model
    └── outputs/                                 meshes / point clouds written by the scripts

The dataset location can be moved without touching the code by setting the
environment variable LINEMESH_DATA (e.g. to a larger disk).
"""
import os
import sys
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- data
DATA_ROOT = Path(os.environ.get("LINEMESH_DATA", REPO_ROOT / "data" / "bridgenet")).expanduser().resolve()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SYNTH_DIR = REPO_ROOT / "data" / "synthetic"

# ----------------------------------------------------------------------------- weights / outputs
WEIGHTS_DIR = REPO_ROOT / "weights"
PRETRAINED_WEIGHTS = WEIGHTS_DIR / "mod_brdgnet_ptrain.pth"     # used by 01_LM_mesh_viz.py
TRAINED_WEIGHTS = WEIGHTS_DIR / "mod_lm_brdnet.pth"             # written by 01_train_LiNeMesh.ipynb
OUTPUT_DIR = REPO_ROOT / "outputs"

# ----------------------------------------------------------------------------- GitHub release assets
GITHUB_REPO = "tmatiki2/LiNeMesh"
RELEASE_BASE = f"https://github.com/{GITHUB_REPO}/releases/download"
WEIGHTS_TAG = "v1.0"                                            # release that holds the pretrained model
WEIGHTS_URL = f"{RELEASE_BASE}/{WEIGHTS_TAG}/{PRETRAINED_WEIGHTS.name}"
DATASET_MANIFEST = REPO_ROOT / "data" / "bridgenet_manifest.json"


def render_dir(split_dir, domain_id):
    """Folder that holds the images, poses, LD and samples of one domain."""
    bid = f"{int(domain_id):05d}"
    return Path(split_dir) / bid / f"{bid}_render"


def list_domains(split_dir):
    """Sorted integer ids of all domains present in a split folder."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    ids = []
    for d in split_dir.iterdir():
        if d.is_dir() and d.name.isdigit() and (d / f"{d.name}_render").is_dir():
            ids.append(int(d.name))
    return sorted(ids)


def download(url, dest, desc=None):
    """Download url to dest with a simple progress bar (standard library only)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    desc = desc or dest.name
    print(f"Downloading {desc}\n  from {url}")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(f"\r  {done / 2**20:8.1f} / {total / 2**20:.1f} MB ({pct:5.1f} %)")
                else:
                    sys.stdout.write(f"\r  {done / 2**20:8.1f} MB")
                sys.stdout.flush()
        print()
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    shutil.move(str(tmp), str(dest))
    return dest


def ensure_weights(path=PRETRAINED_WEIGHTS, url=WEIGHTS_URL):
    """Return the path of the pretrained weights, downloading them on first use."""
    path = Path(path)
    if path.exists():
        return path
    try:
        download(url, path, desc="pretrained LiNeMesh weights")
    except Exception as e:
        raise SystemExit(
            f"\nCould not download the pretrained weights ({e}).\n"
            f"Download '{path.name}' manually from https://github.com/{GITHUB_REPO}/releases\n"
            f"and place it at: {path}\n")
    return path
