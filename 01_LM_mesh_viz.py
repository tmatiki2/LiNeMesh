"""
LiNeMesh evaluation: generate the analysis mesh of a BridgeNet test domain with the
pretrained model, assemble it as an FE mesh (unique nodes + 8-node hexahedra), report the
metrics of the paper against the ground truth and visualise it.

    python 01_LM_mesh_viz.py                    # one of the five test bridges of the paper
    python 01_LM_mesh_viz.py --sample 5015      # a specific test domain
    python 01_LM_mesh_viz.py --no-vis           # no window (servers / SSH); the mesh is still saved

Paths come from LM_paths.py: test data in data/bridgenet/test, weights in weights/,
results in outputs/. The pretrained weights are downloaded automatically on first run.

Outputs (outputs/):
    linemesh_<id>.vtk              hexahedral mesh with the occupancy confidence per element (ParaView)
    linemesh_<id>.inp              same mesh as Abaqus C3D8 elements (nodes/elements only; materials,
                                   supports and loads are defined in the FE software)
    linemesh_<id>_nodes.npy        (M, 3) unique nodes
    linemesh_<id>_elements.npy     (K, 8) connectivity (VTK / C3D8 vertex order)
    linemesh_<id>_mesh_points.npy  raw (N*9, 3) output: centroid + 8 corners per element
    linemesh_<id>_metrics.json     metrics, mesh statistics and the settings of the run
"""
import json
import os
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from LM_models import *
from LM_utils import *
from LM_paths import TEST_DIR, OUTPUT_DIR, PRETRAINED_WEIGHTS, TEST_IDS_FILE, render_dir, list_domains, ensure_weights

# held-out test domains (data/bridgenet_test_ids.json): the five reported in Table II of the
# paper, and all test domains that include a ground-truth point cloud (dense_pts.npy)
_ids = json.loads(TEST_IDS_FILE.read_text())
PAPER_SAMPLES = _ids["paper"]
GT_SAMPLES = _ids["with_ground_truth"]

ap = argparse.ArgumentParser(description="LiNeMesh evaluation on a BridgeNet test domain")
ap.add_argument("--sample", type=int, default=None, help="test domain id (default: one of the five test bridges of the paper)")
ap.add_argument("--wireframe", action="store_true", help="also draw the element edges")
ap.add_argument("--n-views", type=int, default=25, help="number of field images used (default 25)")
ap.add_argument("--mesh-size", type=float, default=0.05, help="element size of the coarse mesh (default 0.05)")
ap.add_argument("--weights", type=str, default=str(PRETRAINED_WEIGHTS), help="model weights (.pth)")
ap.add_argument("--data", type=str, default=str(TEST_DIR), help="folder with the test domains")
ap.add_argument("--no-vis", action="store_true", help="do not open the Open3D window")
ap.add_argument("--seed", type=int, default=0, help="random seed for the selection of the field images (default 0)")
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

# ----------------------------------------------------------------------------- model
weights = ensure_weights(args.weights) if args.weights == str(PRETRAINED_WEIGHTS) else args.weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mod1 = group_occup_sdf().to(device)
res = mod1.load_state_dict(torch.load(weights, map_location=device), strict=False)
if res.missing_keys or res.unexpected_keys:
    print("warning: weights do not match the model exactly:",
          f"{len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")

# ----------------------------------------------------------------------------- test domain
available = list_domains(args.data)
if not available:
    raise SystemExit(f"No test domains found in {args.data}.\n"
                     "Download BridgeNet first:  python scripts/download_bridgenet.py --split test")
if args.sample is not None:
    dmm = args.sample
else:
    cands = [s for s in PAPER_SAMPLES if s in available] or [s for s in GT_SAMPLES if s in available] or available
    dmm = random.choice(cands)
if dmm not in available:
    raise SystemExit(f"Test domain {dmm:05d} not found in {args.data}")
bridge_id = f"{dmm:05d}"
adirs = str(render_dir(args.data, dmm)) + os.sep
print("test domain:", bridge_id)

n_imgs = len([f for f in os.listdir(adirs) if f.startswith("img_") and f.endswith(".jpg")])
nt = min(args.n_views, n_imgs)
inp_idx = sorted(random.sample(range(n_imgs), nt))
print(f"field images ({nt} of {n_imgs}, seed {args.seed}):", inp_idx)
img_s = [cv2.imread(adirs + 'img_' + str(jj) + '.jpg') for jj in inp_idx]
lnb = np.load(adirs + 'line_bars.npy').astype(np.float64)   # stored as float16; the coarse mesh is built in float64
gt_file = adirs + 'dense_pts.npy'
xyz_gt = np.load(gt_file).astype(np.float64) if os.path.exists(gt_file) else None
if xyz_gt is None:
    print("no ground truth (dense_pts.npy) for this domain: the mesh is generated but not scored")

## --- create coarse mesh ---
lnbm = lnb.reshape(-1, 3)
mnx = np.min(lnbm, 0)
mxx = np.max(lnbm, 0)
ms = args.mesh_size  # --- mesh size
xr = np.arange(mnx[0] - 0.5, mxx[0] + 0.5, ms)
yr = np.arange(mnx[1] - 0.5, mxx[1] + 0.5, ms)
zr = np.arange(mnx[2] - 0.5, mxx[2] + 0.5, ms)
xrr, yrr, zrr = np.meshgrid(xr, yr, zr)
xrr, yrr, zrr = xrr.flatten(), yrr.flatten(), zrr.flatten()
amesh = np.vstack((xrr, yrr, zrr)).T
hv = ms * np.ones(len(amesh)).reshape(-1, 1)
amesh = c2verts3(amesh, hv, szb=ms, fix=True)
print("mesh_size: ", amesh.shape)

## -- preprocess canonical and field images
all_inps = []
sz = 224
for idx in range(len(img_s)):
    img1_pz = np.array(img_s[idx]).astype(np.uint8)
    img_tsr1 = F.interpolate(torch.tensor(2 * (img1_pz / 255.0) - 1.0).permute(2, 0, 1).unsqueeze(0), size=(sz, sz),
                             mode='bilinear', align_corners=False)
    all_inps.append(img_tsr1)
imgs_f = torch.concat(all_inps, 0)
gimgs, imgs, Ts, Rs = gen_cano_imgs_only(lnb, rad=1, rmr=False)

## Run evals and keep the occupied elements
pass_meshes = []
pass_conf = []
nmax = 90000  # number of mesh points (10,000 elements) per batch
with torch.no_grad():
    mod1n = mod1.eval()
    for start in range(0, len(amesh), nmax):
        end = min(start + nmax, len(amesh))
        meshs = np.array(amesh[start:end, :])
        out2 = data_eval(meshs, Ts, Rs, imgs, imgs_f)
        all_inps, _, vis_mat_p = out2
        pred_cls_ = mod1n([all_inps], [vis_mat_p])
        # the head is trained with labels {0: empty, 1: occupied}; an element is kept when its
        # predicted occupancy probability is at least 50 % (Section 3.6 of the paper)
        prob = torch.softmax(pred_cls_.cpu()[:, :3], 1).numpy()
        occ = prob[:, 1] >= 0.5
        if occ.any():
            pass_meshes.append(meshs.reshape(-1, 9, 3)[occ].reshape(-1, 3))
            pass_conf.append(prob[occ, 1])
        print(f"processed {end}/{len(amesh)} mesh points")

if not pass_meshes:
    raise SystemExit("No mesh elements were classified as occupied.")
npass_meshes = np.vstack(pass_meshes)      # (N*9, 3): centroid + 8 corners per occupied element
conf = np.concatenate(pass_conf)           # occupancy confidence of each occupied element

## --- FE mesh: unique nodes (closer than 1e-4 merged) and hexahedral connectivity.
# The candidate cubes evaluated by the network (half-size = element size) overlap their
# neighbours; element_size rebuilds each occupied element as a cube of the coarse-mesh size
# about its centroid, so neighbouring elements share nodes and faces (a conforming mesh).
nodes, elements, keep, info = points_to_fe_mesh(npass_meshes, tol=1e-4, element_size=ms, return_info=True)
conf = conf[keep]
print("FE mesh: {nodes} nodes, {elements_out} hexahedra, {connected_components} connected component(s), "
      "{boundary_faces} boundary faces".format(**info))

## --- metrics (paper protocol): unique mesh points against the ground truth, at 5 and 10 cm
eval_pts = unique_points(npass_meshes, tol=1e-4)
metrics = {}
if xyz_gt is not None:
    for tau in (0.05, 0.10):
        print("----- metrics @ tau = %.2f m -----" % tau)
        m = pointcloud_metrics(eval_pts, xyz_gt, threshold=tau)
        metrics[f"{tau:.2f}"] = {k: v for k, v in m.items() if not k.endswith("distances")}

## --- save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stem = OUTPUT_DIR / f"linemesh_{bridge_id}"
save_fe_mesh(f"{stem}.vtk", nodes, elements, cell_data={"occupancy_confidence": conf})
save_fe_mesh(f"{stem}.inp", nodes, elements)
np.save(f"{stem}_nodes.npy", nodes)
np.save(f"{stem}_elements.npy", elements)
np.save(f"{stem}_mesh_points.npy", npass_meshes.astype(np.float32))
run = dict(domain=bridge_id, seed=args.seed, n_views=nt, field_images=inp_idx, mesh_size=ms,
           weights=str(weights), candidate_elements=len(amesh) // 9, evaluated_points=len(eval_pts),
           fe_mesh=info, metrics=metrics)
with open(f"{stem}_metrics.json", "w") as f:
    json.dump(run, f, indent=1, default=float)
print("saved:", f"{stem}.vtk", f"{stem}.inp")

## --- visualise the boundary faces of the mesh, coloured by occupancy confidence
if not args.no_vis:
    import open3d as o3d
    import matplotlib.cm as cm
    faces, owner, count = hex_faces(elements)
    bnd = count == 1
    quads, own = faces[bnd], owner[bnd]
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # per-vertex colour: duplicate the face vertices so each face carries its element's colour
    c = cm.viridis(np.clip((conf - 0.5) / 0.5, 0, 1))[:, :3]
    vtx = nodes[tris].reshape(-1, 3)
    col = np.repeat(np.vstack([c[own], c[own]]), 3, axis=0)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vtx),
                                     o3d.utility.Vector3iVector(np.arange(len(vtx)).reshape(-1, 3)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    mesh.compute_vertex_normals()
    geoms = [mesh]
    if args.wireframe:
        edges = np.unique(np.sort(np.vstack([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]]), 1), axis=0)
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(nodes), o3d.utility.Vector2iVector(edges))
        ls.paint_uniform_color([0.1, 0.1, 0.1])
        geoms.append(ls)
    o3d.visualization.draw_geometries(geoms, window_name=f"LiNeMesh {bridge_id}: analysis mesh "
                                      "(colour: occupancy confidence 0.5 -> 1)", width=1000, height=700)
