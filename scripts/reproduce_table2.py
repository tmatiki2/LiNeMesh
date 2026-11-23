"""
Run the pretrained LiNeMesh on the five BridgeNet test bridges of Table II of the paper and
print the LiNeMesh rows (precision, recall, F1, IoU at 5 and 10 cm, mean accuracy).

    python scripts/reproduce_table2.py              # all five bridges, seed 0
    python scripts/reproduce_table2.py --seed 1     # another random selection of field images

The values of the paper are printed below the new ones. Differences of a few hundredths are
expected: the 25 field images are drawn at random (here with a fixed seed), and the coarse mesh
is built at exactly 5 cm (the runs of the paper used a slightly different, reduced-precision grid).
Each run writes outputs/linemesh_<id>.* (see 01_LM_mesh_viz.py); the table is also saved to
outputs/table2_linemesh.md. Requires the test split (python scripts/download_bridgenet.py --split test).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from LM_paths import OUTPUT_DIR, TEST_IDS_FILE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-views", type=int, default=25)
ap.add_argument("--ids", type=int, nargs="+", default=None, help="test ids (default: the five of the paper)")
args = ap.parse_args()
ids = args.ids or json.loads(TEST_IDS_FILE.read_text())["paper"]

rows = []
for i in ids:
    cmd = [sys.executable, str(ROOT / "01_LM_mesh_viz.py"), "--sample", str(i), "--no-vis",
           "--seed", str(args.seed), "--n-views", str(args.n_views)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    r = json.loads((OUTPUT_DIR / f"linemesh_{i:05d}_metrics.json").read_text())
    m5, m10 = r["metrics"]["0.05"], r["metrics"]["0.10"]
    rows.append((f"{i:05d}", m5, m10, r["fe_mesh"]))

hdr = ("| Bridge | P 5cm | P 10cm | R 5cm | R 10cm | F1 5cm | F1 10cm | IoU 5cm | IoU 10cm | A (cm) | elements | nodes |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
PAPER = {"05015": (0.375, 0.805, 0.969, 0.972, 0.541, 0.881, 0.183, 0.327, 6.83),
         "05209": (0.315, 0.737, 0.887, 0.976, 0.465, 0.840, 0.114, 0.390, 8.76),
         "05311": (0.373, 0.711, 0.968, 0.974, 0.538, 0.822, 0.202, 0.304, 8.14),
         "05328": (0.305, 0.570, 0.943, 0.963, 0.461, 0.716, 0.200, 0.290, 13.20),
         "05384": (0.287, 0.620, 0.919, 0.924, 0.438, 0.742, 0.171, 0.266, 10.60)}
lines = [f"| {b} | {a['precision']:.3f} | {c['precision']:.3f} | {a['recall']:.3f} | {c['recall']:.3f} | "
         f"{a['f1']:.3f} | {c['f1']:.3f} | {a['iou']:.3f} | {c['iou']:.3f} | {100 * a['accuracy']:.2f} | "
         f"{fe['elements_out']} | {fe['nodes']} |" for b, a, c, fe in rows]
lines += [f"| {b} (paper) | " + " | ".join(f"{v:.3f}" for v in p[:8]) + f" | {p[8]:.2f} | | |"
          for b, p in PAPER.items() if b in [r[0] for r in rows]]
table = hdr + "\n".join(lines) + "\n"
print("\n" + table)
(OUTPUT_DIR / "table2_linemesh.md").write_text(table)
