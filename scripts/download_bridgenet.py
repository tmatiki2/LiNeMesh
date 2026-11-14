"""
Download and unpack the BridgeNet dataset into data/bridgenet/ (or $LINEMESH_DATA).

    python scripts/download_bridgenet.py               # train + test
    python scripts/download_bridgenet.py --split test  # test only (enough for 01_LM_mesh_viz.py)
    python scripts/download_bridgenet.py --keep-zip    # keep the downloaded archives

The list of archives is read from data/bridgenet_manifest.json (GitHub release assets).
Large splits may be published as several parts (e.g. bridgenet_train_part01.zip, ...);
all parts listed in the manifest are downloaded and merged.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LM_paths import DATA_ROOT, DATASET_MANIFEST, RELEASE_BASE, GITHUB_REPO, download, list_domains


def merge_tree(src, dst):
    """Move the contents of src into dst (merging folders)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir() and target.exists():
            merge_tree(item, target)
            shutil.rmtree(item, ignore_errors=True)
        elif not target.exists():
            shutil.move(str(item), str(target))


def extract(zip_path, split):
    """Extract an archive so that its domains end up in DATA_ROOT/<split>/."""
    tmp = DATA_ROOT / "_extract_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    # locate the folder that directly contains the domain folders (<id>/<id>_render)
    root = None
    for cand in [tmp] + sorted(p for p in tmp.rglob("*") if p.is_dir()):
        if list_domains(cand):
            root = cand
            break
    if root is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{zip_path.name}: no <id>/<id>_render folders found")
    merge_tree(root, DATA_ROOT / split)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Download BridgeNet")
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--keep-zip", action="store_true", help="keep the downloaded archives in data/downloads")
    args = ap.parse_args()

    man = json.loads(DATASET_MANIFEST.read_text())
    files = [f for f in man["files"] if args.split == "all" or f["split"] == args.split]
    dl_dir = DATA_ROOT.parent / "downloads"
    for f in files:
        url = f.get("url") or f"{RELEASE_BASE}/{man['tag']}/{f['name']}"
        zpath = dl_dir / f["name"]
        if not zpath.exists():
            try:
                download(url, zpath)
            except Exception as e:
                raise SystemExit(
                    f"\nCould not download {f['name']} ({e}).\n"
                    f"If the dataset has not been released yet, check https://github.com/{GITHUB_REPO}/releases\n"
                    f"or download the archives manually into {dl_dir} and run this script again.")
        extract(zpath, f["split"])
        if not args.keep_zip:
            zpath.unlink()
    for split in ("train", "test"):
        n = len(list_domains(DATA_ROOT / split))
        if n:
            print(f"{split}: {n} domains in {DATA_ROOT / split}")
    print("Done.")


if __name__ == "__main__":
    main()
