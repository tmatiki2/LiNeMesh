"""
Run the pretrained LiNeMesh on the five BridgeNet test bridges of Table II of the paper and
print the LiNeMesh rows (precision, recall, F1, IoU at 5 and 10 cm, mean accuracy).

    python scripts/reproduce_table2.py              # all five bridges, seed 0
    python scripts/reproduce_table2.py --seed 1     # another random selection of field images

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
lines = [f"| {b} | {a['precision']:.3f} | {c['precision']:.3f} | {a['recall']:.3f} | {c['recall']:.3f} | "
         f"{a['f1']:.3f} | {c['f1']:.3f} | {a['iou']:.3f} | {c['iou']:.3f} | {100 * a['accuracy']:.2f} | "
         f"{fe['elements_out']} | {fe['nodes']} |" for b, a, c, fe in rows]
table = hdr + "\n".join(lines) + "\n"
print("\n" + table)
(OUTPUT_DIR / "table2_linemesh.md").write_text(table)
