"""
Convert a saved LiNeMesh output, an (N*9, 3) array of mesh points (centroid + 8 corners per
element), into an FE mesh and, optionally, evaluate it against a reference point cloud.

    python scripts/mesh_from_points.py outputs/linemesh_05015_mesh_points.npy
    python scripts/mesh_from_points.py my_points.npy --ref reference.npy --tau 0.05 0.10
    python scripts/mesh_from_points.py my_points.npy --element-size 0.05 --out outputs/my_mesh

Writes <out>.vtk (ParaView), <out>.inp (Abaqus C3D8), <out>_nodes.npy and <out>_elements.npy.
Metrics (paper protocol): precision, recall, F1, mean accuracy and volumetric IoU of the unique
mesh points (points closer than --tol merged) against the reference, at each threshold tau.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LM_utils import points_to_fe_mesh, save_fe_mesh, unique_points, pointcloud_metrics  # noqa: E402

ap = argparse.ArgumentParser(description="LiNeMesh output (N*9, 3) -> FE mesh (+ metrics)")
ap.add_argument("points", help=".npy file with the (N*9, 3) mesh points")
ap.add_argument("--ref", default=None, help="reference point cloud (.npy, (M, 3)) for the metrics")
ap.add_argument("--tau", type=float, nargs="+", default=[0.05, 0.10], help="distance thresholds (default 0.05 0.10)")
ap.add_argument("--tol", type=float, default=1e-4, help="merge tolerance for coincident nodes (default 1e-4)")
ap.add_argument("--element-size", default="auto",
                help='edge length of the rebuilt elements: "auto" (coarse-mesh spacing, default), '
                     'a number, or "none" to keep the stored corners')
ap.add_argument("--out", default=None, help="output stem (default: next to the input file)")
args = ap.parse_args()

pts = np.load(args.points)[:, :3].astype(np.float64)
if len(pts) % 9:
    raise SystemExit(f"{args.points}: {len(pts)} points is not a multiple of 9")
es = args.element_size.lower()
es = None if es == "none" else ("auto" if es == "auto" else float(es))

nodes, elements, keep, info = points_to_fe_mesh(pts, tol=args.tol, element_size=es, return_info=True)
for k, v in info.items():
    print(f"{k:>22s}: {v}")

stem = Path(args.out) if args.out else Path(args.points).with_suffix("")
stem.parent.mkdir(parents=True, exist_ok=True)
save_fe_mesh(f"{stem}.vtk", nodes, elements)
save_fe_mesh(f"{stem}.inp", nodes, elements)
np.save(f"{stem}_nodes.npy", nodes)
np.save(f"{stem}_elements.npy", elements)
print("saved:", f"{stem}.vtk", f"{stem}.inp")

if args.ref:
    ref = np.load(args.ref)[:, :3].astype(np.float64)
    ev = unique_points(pts, tol=args.tol)
    print(f"evaluated points: {len(ev)} unique of {len(pts)}")
    for tau in args.tau:
        print(f"----- metrics @ tau = {tau:g} -----")
        pointcloud_metrics(ev, ref, threshold=tau)
