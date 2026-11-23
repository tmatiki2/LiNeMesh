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
# LINEMESH_RELEASE_BASE can point to a mirror holding the same release assets
RELEASE_BASE = os.environ.get("LINEMESH_RELEASE_BASE", f"https://github.com/{GITHUB_REPO}/releases/download")
WEIGHTS_TAG = "v1.0"                                            # release that holds the pretrained model
WEIGHTS_ZIP = PRETRAINED_WEIGHTS.with_suffix(".zip").name               # release asset: zip holding the .pth
WEIGHTS_URL = f"{RELEASE_BASE}/{WEIGHTS_TAG}/{WEIGHTS_ZIP}"
DATASET_MANIFEST = REPO_ROOT / "data" / "bridgenet_manifest.json"
TEST_IDS_FILE = REPO_ROOT / "data" / "bridgenet_test_ids.json"   # paper test ids and ids with ground truth


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


def download(url, dest, desc=None, expected_size=None):
    """Download url to dest with a progress bar (standard library only).
    An interrupted download (dest + '.part') is resumed with an HTTP Range request.
    If expected_size is given, the final size is checked."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    desc = desc or dest.name
    done = tmp.stat().st_size if tmp.exists() else 0
    if expected_size is not None and done > expected_size:
        tmp.unlink(); done = 0
    print(f"Downloading {desc}\n  from {url}" + (f"\n  resuming at {done / 2**20:.1f} MB" if done else ""))
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r:
        if done and r.status != 206:          # server ignored the range: start again
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(tmp, "ab" if done else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 2**20:9.1f} / {total / 2**20:.1f} MB ({100 * done / total:5.1f} %)")
                else:
                    sys.stdout.write(f"\r  {done / 2**20:9.1f} MB")
                sys.stdout.flush()
    print()
    if expected_size is not None and done != expected_size:
        raise IOError(f"{desc}: downloaded {done} bytes, expected {expected_size} (run again to resume)")
    shutil.move(str(tmp), str(dest))
    return dest


def ensure_weights(path=PRETRAINED_WEIGHTS, url=WEIGHTS_URL):
    """Return the path of the pretrained weights, downloading and unzipping them on first use
    (release asset mod_brdgnet_ptrain.zip, ~1.9 GB, holding mod_brdgnet_ptrain.pth)."""
    import zipfile
    path = Path(path)
    if path.exists():
        return path
    zpath = path.with_suffix(".zip")
    try:
        download(url, zpath, desc="pretrained LiNeMesh weights")
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith(".pth"))
            print(f"Extracting {member} ...")
            with z.open(member) as src, open(path, "wb") as dst:   # CRC-32 checked by zipfile
                shutil.copyfileobj(src, dst, 1 << 20)
        zpath.unlink()
    except Exception as e:
        raise SystemExit(
            f"\nCould not obtain the pretrained weights ({e}).\n"
            f"Download '{zpath.name}' from https://github.com/{GITHUB_REPO}/releases/tag/{WEIGHTS_TAG},\n"
            f"unzip it and place '{path.name}' at: {path}\n")
    return path
