"""
LiNeMesh evaluation: generate an analysis mesh for a BridgeNet test domain with the
pretrained model, report point-cloud metrics against the ground truth and visualise it.

    python 01_LM_mesh_viz.py                    # random test domain from the paper's list
    python 01_LM_mesh_viz.py --sample 5040      # a specific test domain
    python 01_LM_mesh_viz.py --no-vis           # no window (servers / SSH); the point cloud is still saved

Paths come from LM_paths.py: test data in data/bridgenet/test, weights in weights/,
results in outputs/. The pretrained weights are downloaded automatically on first run.
"""
import os
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from LM_models import *
from LM_utils import *
from LM_paths import TEST_DIR, OUTPUT_DIR, PRETRAINED_WEIGHTS, render_dir, list_domains, ensure_weights

# test domains evaluated in the paper
PAPER_SAMPLES = [5040, 5045, 5015, 5311, 5384, 5226, 5331, 5209, 5328, 5278, 5258, 5145, 5462, 5262, 5269,
                 5408, 5493, 5491, 5263, 5028, 5480]

ap = argparse.ArgumentParser(description="LiNeMesh evaluation on a BridgeNet test domain")
ap.add_argument("--sample", type=int, default=None, help="test domain id (default: random paper sample)")
ap.add_argument("--n-views", type=int, default=25, help="number of field images used (default 25)")
ap.add_argument("--mesh-size", type=float, default=0.05, help="element size of the coarse mesh (default 0.05)")
ap.add_argument("--weights", type=str, default=str(PRETRAINED_WEIGHTS), help="model weights (.pth)")
ap.add_argument("--data", type=str, default=str(TEST_DIR), help="folder with the test domains")
ap.add_argument("--no-vis", action="store_true", help="do not open the Open3D window")
ap.add_argument("--seed", type=int, default=None, help="random seed")
args = ap.parse_args()

if args.seed is not None:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

# ----------------------------------------------------------------------------- model
weights = ensure_weights(args.weights) if args.weights == str(PRETRAINED_WEIGHTS) else args.weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mod1 = group_occup_sdf().to(device)
mod1.load_state_dict(torch.load(weights, map_location=device), strict=False)

# ----------------------------------------------------------------------------- test domain
available = list_domains(args.data)
if not available:
    raise SystemExit(f"No test domains found in {args.data}.\n"
                     "Download BridgeNet first:  python scripts/download_bridgenet.py --split test")
if args.sample is not None:
    dmm = args.sample
else:
    cands = [s for s in PAPER_SAMPLES if s in available] or available
    dmm = random.choice(cands)
if dmm not in available:
    raise SystemExit(f"Test domain {dmm:05d} not found in {args.data}")
bridge_id = f"{dmm:05d}"
adirs = str(render_dir(args.data, dmm)) + os.sep
print("test domain:", bridge_id)

n_imgs = len([f for f in os.listdir(adirs) if f.startswith("img_") and f.endswith(".jpg")])
nt = min(args.n_views, n_imgs)
inp_idx = random.sample(range(n_imgs), nt)
img_s = [cv2.imread(adirs + 'img_' + str(jj) + '.jpg') for jj in inp_idx]
lnb = np.load(adirs + 'line_bars.npy')
xyz_gt = np.load(adirs + 'dense_pts.npy')

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

## Run evals and build meshes
pass_meshes = []
pass_cols = []
nmax = 90000
with torch.no_grad():
    mod1n = mod1.eval()
    for start in range(0, len(amesh), nmax):
        end = min(start + nmax, len(amesh))
        meshs = np.array(amesh[start:end, :])
        out2 = data_eval(meshs, Ts, Rs, imgs, imgs_f)
        all_inps, _, vis_mat_p = out2
        pred_cls_ = mod1n([all_inps], [vis_mat_p])
        predds = pred_cls_.cpu()
        pred_cls = torch.argmax(torch.softmax(predds[:, :3], 1), 1).numpy()
        pred_cols = predds[:, 3:6].numpy()
        pps = [[meshs[kkp * 9:(kkp * 9 + 9), :], np.vstack([pred_cols[kkp]] * 9)] for kkp in range(len(pred_cls))
               if pred_cls[kkp] == 1]
        if len(pps) > 0:
            msh, cols = map(list, zip(*pps))
            pass_meshes.append(np.vstack(msh))
            pass_cols.append(np.vstack(cols))
        print(f"processed {end}/{len(amesh)} vertices")

if not pass_meshes:
    raise SystemExit("No mesh elements were classified as occupied.")
npass_meshes = np.vstack(pass_meshes)
cols = 0.6 * np.ones_like(npass_meshes)

## --- compute metrics and visualize mesh vertices
th_mtx1 = 0.05
th_mtx2 = 0.1
print("----- metrics @ tau = %.2f -----" % th_mtx1)
metrics1 = pointcloud_metrics(npass_meshes, xyz_gt, threshold=th_mtx1)
print("----- metrics @ tau = %.2f -----" % th_mtx2)
metrics2 = pointcloud_metrics(npass_meshes, xyz_gt, threshold=th_mtx2)

import open3d as o3d

point_cloud_i = o3d.geometry.PointCloud()
point_cloud_i.points = o3d.utility.Vector3dVector(npass_meshes)
point_cloud_i.colors = o3d.utility.Vector3dVector(cols)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_ply = OUTPUT_DIR / f"linemesh_{bridge_id}.ply"
o3d.io.write_point_cloud(str(out_ply), point_cloud_i)
np.save(OUTPUT_DIR / f"linemesh_{bridge_id}_mesh_vertices.npy", npass_meshes.astype(np.float32))
print("saved:", out_ply)

if not args.no_vis:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='LiNeMesh: mesh vertices', width=800, height=600, left=50, top=50)
    vis.add_geometry(point_cloud_i)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.point_show_normal = False
    vis.run()
    vis.destroy_window()
