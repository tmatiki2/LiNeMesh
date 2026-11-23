"""
Download the BridgeNet synthetic dataset and unpack it into data/bridgenet/ (or $LINEMESH_DATA).

BridgeNet is published on the GitHub release v1.0 as one ZIP archive split into twelve binary
chunks, part_000 ... part_011, plus original.sha256 (SHA-256 of the complete archive):

    python scripts/download_bridgenet.py                # train + test (~22.3 GB download)
    python scripts/download_bridgenet.py --split test   # test split only (downloads 3 parts, ~4.3 GB)
    python scripts/download_bridgenet.py --split train  # training split only

--split all  downloads every part, reconstructs the archive by concatenating the parts in order
             (identical to:  cat part_000 part_001 ... part_011 > bridgenet.zip), checks its
             SHA-256 against original.sha256 and extracts train/ and test/.
--split test / train  downloads only the parts that hold that split (the archive's central
             directory is in the last part) and extracts it directly from the parts, without
             building the 22 GB archive; every file is checked against its CRC-32 on extraction.

Downloads are resumed if interrupted. Parts downloaded manually can be used with --parts-dir.
Result:
    data/bridgenet/train/<id>/<id>_render/   5000 domains, ids 00000-04999
    data/bridgenet/test/<id>/<id>_render/     500 domains, ids 05000-05499
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LM_paths import DATA_ROOT, DATASET_MANIFEST, RELEASE_BASE, GITHUB_REPO, download, list_domains  # noqa: E402


class MultiPartReader(io.RawIOBase):
    """Read-only, seekable view of the concatenated parts (the archive) without writing it to disk.
    Parts that are not present may not be read."""

    def __init__(self, paths, sizes):
        self.paths, self.sizes = [Path(p) for p in paths], list(sizes)
        self.starts = [0]
        for s in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + s)
        self.total = sum(self.sizes)
        self.pos = 0
        self._files = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def _file(self, k):
        if k not in self._files:
            if not self.paths[k].exists():
                raise IOError(f"{self.paths[k].name} is needed but was not downloaded")
            self._files[k] = open(self.paths[k], "rb")
        return self._files[k]

    def readinto(self, b):
        n, view, got = len(b), memoryview(b), 0
        while got < n and self.pos < self.total:
            k = max(i for i, s in enumerate(self.starts) if s <= self.pos)
            f = self._file(k)
            f.seek(self.pos - self.starts[k])
            chunk = f.read(min(n - got, self.starts[k] + self.sizes[k] - self.pos))
            if not chunk:
                break
            view[got:got + len(chunk)] = chunk
            got += len(chunk); self.pos += len(chunk)
        return got

    def close(self):
        for f in self._files.values():
            f.close()
        super().close()


def concatenate(parts, out):
    """cat part_000 part_001 ... > out (streamed), returning the SHA-256 of the result."""
    h = hashlib.sha256()
    total = sum(p.stat().st_size for p in parts); done = 0
    print(f"Reconstructing {out.name} from {len(parts)} parts (cat) ...")
    with open(out, "wb") as fo:
        for p in parts:
            with open(p, "rb") as fi:
                while True:
                    chunk = fi.read(1 << 24)
                    if not chunk:
                        break
                    fo.write(chunk); h.update(chunk); done += len(chunk)
                    sys.stdout.write(f"\r  {done / 2**30:6.2f} / {total / 2**30:.2f} GB"); sys.stdout.flush()
    print()
    return h.hexdigest()


def extract(zf, root, splits, dest):
    """Extract <root>/<split>/... members of an open ZipFile into dest/<split>/..."""
    members = [m for m in zf.infolist()
               if any(m.filename.startswith(f"{root}/{s}/") for s in splits) and not m.is_dir()]
    if not members:
        raise SystemExit(f"no members under {root}/{{{','.join(splits)}}}/ in the archive")
    total = sum(m.file_size for m in members); done = 0
    print(f"Extracting {len(members)} files ({total / 2**30:.1f} GB) into {dest} ...")
    for i, m in enumerate(members):
        target = dest / m.filename[len(root) + 1:]
        if target.exists() and target.stat().st_size == m.file_size:
            done += m.file_size; continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, open(target, "wb") as dst:   # CRC-32 checked by zipfile
            shutil.copyfileobj(src, dst, 1 << 20)
        done += m.file_size
        if i % 500 == 0 or i == len(members) - 1:
            sys.stdout.write(f"\r  {i + 1}/{len(members)} files, {done / 2**30:.2f} GB"); sys.stdout.flush()
    print()


def needed_parts(zf_reader_factory, man, root, splits):
    """Indices of the parts that hold the members of the requested splits (plus the last part)."""
    with zipfile.ZipFile(zf_reader_factory()) as zf:
        sel = [m for m in zf.infolist() if any(m.filename.startswith(f"{root}/{s}/") for s in splits)]
    lo = min(m.header_offset for m in sel)
    hi = max(m.header_offset + 30 + len(m.filename.encode()) + 256 + m.compress_size for m in sel)
    starts, k = [], 0
    for p in man["parts"]:
        starts.append(k); k += p["size"]
    idx = {i for i, s in enumerate(starts) if s < hi and s + man["parts"][i]["size"] > lo}
    return sorted(idx | {len(starts) - 1})


def main():
    ap = argparse.ArgumentParser(description="Download BridgeNet (GitHub release parts)")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--parts-dir", default=None, help="folder for the downloaded parts (default data/downloads)")
    ap.add_argument("--keep-parts", action="store_true", help="keep the parts after extraction")
    ap.add_argument("--keep-zip", action="store_true", help="--split all: keep the reconstructed bridgenet.zip")
    args = ap.parse_args()

    man = json.loads(DATASET_MANIFEST.read_text())
    root = man["archive_root"]
    pdir = Path(args.parts_dir) if args.parts_dir else DATA_ROOT.parent / "downloads"
    pdir.mkdir(parents=True, exist_ok=True)
    paths = [pdir / p["name"] for p in man["parts"]]
    sizes = [p["size"] for p in man["parts"]]
    base = f"{RELEASE_BASE}/{man['tag']}"

    def fetch(i):
        p = man["parts"][i]
        try:
            download(f"{base}/{p['name']}", paths[i], desc=f"{p['name']} ({i + 1}/{len(paths)})", expected_size=p["size"])
        except Exception as e:
            raise SystemExit(f"\nCould not download {p['name']} ({e}).\nRun the script again to resume, or download "
                             f"the parts from https://github.com/{GITHUB_REPO}/releases/tag/{man['tag']} into {pdir}\n"
                             f"and run:  python scripts/download_bridgenet.py --parts-dir {pdir}")

    free = shutil.disk_usage(pdir).free / 2**30
    print(f"Free disk space at {pdir}: {free:.1f} GB")

    if args.split == "all":
        need = man["archive_size"] * 2 / 2**30 + 50
        if free < need:
            print(f"Warning: --split all needs about {need:.0f} GB (parts + archive + extracted data).")
        for i in range(len(paths)):
            fetch(i)
        expected = man["sha256"]
        try:
            with urllib.request.urlopen(f"{base}/{man['checksum_asset']}") as r:
                expected = r.read().decode().split()[0].strip().lower()
        except Exception:
            print("(using the SHA-256 from the manifest)")
        zpath = pdir / man["archive"]
        digest = concatenate(paths, zpath)
        if digest != expected:
            zpath.unlink()
            raise SystemExit(f"SHA-256 mismatch: {digest} != {expected}. Delete the parts in {pdir} and download again.")
        print("SHA-256 verified:", digest)
        if not args.keep_parts:
            for p in paths:
                p.unlink()
        with zipfile.ZipFile(zpath) as zf:
            extract(zf, root, ["train", "test"], DATA_ROOT)
        if not args.keep_zip:
            zpath.unlink()
    else:
        fetch(len(paths) - 1)   # the last part holds the central directory
        parts = needed_parts(lambda: MultiPartReader(paths, sizes), man, root, [args.split])
        print(f"{args.split} split: parts {', '.join(man['parts'][i]['name'] for i in parts)}")
        for i in parts:
            fetch(i)
        reader = MultiPartReader(paths, sizes)
        with zipfile.ZipFile(reader) as zf:
            extract(zf, root, [args.split], DATA_ROOT)
        reader.close()
        if not args.keep_parts:
            for i in parts:
                paths[i].unlink()

    for split in ("train", "test"):
        n = len(list_domains(DATA_ROOT / split))
        if n:
            print(f"{split}: {n} domains in {DATA_ROOT / split}")
    print("Done.")


if __name__ == "__main__":
    main()
