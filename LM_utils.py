import torch
import numpy as np
import time
import cv2
import time
import random
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
roll_c_u, pitch_c_u, yaw_c_u = -(3.141592/2), 0, 3.141592/2 #r_xr, r_yr, r_zr#  
rx = np.matrix([[1, 0, 0],[0, math.cos(roll_c_u),-math.sin(roll_c_u)], [0, math.sin(roll_c_u), math.cos(roll_c_u)]])
ry = np.matrix([[math.cos(pitch_c_u), 0, math.sin(pitch_c_u)],[0, 1, 0],[-math.sin(pitch_c_u), 0, math.cos(pitch_c_u)]])
rz = np.matrix([[math.cos(yaw_c_u), -math.sin(yaw_c_u), 0],[math.sin(yaw_c_u), math.cos(yaw_c_u), 0],[0, 0, 1]])
R_c_b = np.matmul(np.matmul(rz,ry),rx)

K_mm_r = np.array([[385.464,0,330.348],[0,385.093,241.54],[0, 0, 1]])
sxc = 300/640 #224/640 #
syc = 300/480 #224/480 #
kfr = np.array([[sxc,0,sxc],[0,syc,syc],[0,0,1]])
K_mm_rs = kfr * K_mm_r


def c2verts(centroids, h=None, szb=0.15, fix=False):
    N = centroids.shape[0]
    if fix:
        h = np.random.uniform(szb, szb, size=(N, 1))
    elif h is None:
        h = np.random.uniform(0.001, szb, size=(N, 1))  # (N, 1)
    base_offsets = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ])  # (8, 3)
    offsets = base_offsets[None, :, :] * h[:, None, :]  # (N, 8, 3)
    vertices = centroids[:, None, :] + offsets  # (N, 8, 3)
    centroids_expanded = centroids[:, None, :]  # (N, 1, 3)
    all_points = np.concatenate([centroids_expanded, vertices], axis=1)  # (N, 9, 3)
    # oidx, iddx = g_shuffle(N)
    return all_points.reshape(-1, 3)#, oidx, iddx

def c2verts3(centroids, h=None, szb=0.15, fix=False):

    N = centroids.shape[0]

    if fix:
        h = np.full((N, 1), szb, dtype=np.float64)

    elif h is None:
        h = np.random.uniform(
            0.001,
            szb,
            size=(N, 1)
        )

    else:
        h = np.asarray(h, dtype=np.float64)

        if h.ndim == 0:
            h = np.full((N, 1), h)

        elif h.ndim == 1:
            h = h[:, None]

    # PyVista-compatible hexahedron ordering
    base_offsets = np.array([
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ], dtype=np.float64)

    # (N, 8, 3)
    offsets = base_offsets[None, :, :] * h[:, None, :]

    # (N, 8, 3)
    vertices = centroids[:, None, :] + offsets

    # Centroid + 8 vertices
    # (N, 9, 3)
    centroids_expanded = centroids[:, None, :]

    all_points = np.concatenate(
        [centroids_expanded, vertices],
        axis=1
    )

    # (N*9, 3), same format as the original function
    return all_points.reshape(-1, 3)

def proj_3d2d(xyz_ins, t1p, Rgc, wi=300, hi=300):
    uvs = K_mm_rs@Rgc@(xyz_ins.T - t1p.reshape(-1,1))
    uv = uvs[:2,:] / uvs[2:,:]
    uv = uv.T
    ff = uv[:,0]
    ff = (uv[:,0]>=0) * (uv[:,0]<wi) * (uv[:,1]>=0) * (uv[:,1]<hi) * (uvs[2,:]>0.1)
    return uv, ff

def draw_shape(img, Km,tm,Rm,lnb,color=(255,0,0),thickness=1):
    imgn = np.array(img).astype(np.uint8)
    lnb_ = np.array(lnb)
    for kk in range(lnb_.shape[0]):
        xyz = lnb_[kk]
        uvz = Km@Rm@(xyz.T - tm.reshape(-1,1))
        uv = uvz[:2,:] / uvz[2:,:]
        u,v = uv[0,:].astype(np.int16), uv[1,:].astype(np.int16)
        pt1, pt2 = (u[0],v[0]), (u[1],v[1])
        cv2.line(imgn, pt1, pt2, color, thickness)
    return imgn

def get_camera_rotation(camera_xyz, target_xyz):
    camera_xyz = np.asarray(camera_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    direction = target_xyz - camera_xyz
    direction /= np.linalg.norm(direction)
    z_cam = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(world_up, z_cam)
    if np.linalg.norm(x_cam) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(world_up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= np.linalg.norm(y_cam)

    R_cam2world = np.column_stack([
        x_cam,
        y_cam,
        z_cam
    ])
    R_world2cam_blender = R_cam2world.T
    R_bb_c = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    R_world2cam = R_bb_c @ R_world2cam_blender

    R_b_g = R_world2cam.T @ R_c_b.T
    r_b_g = R.from_matrix(R_b_g).as_euler('xyz',degrees=True)

    return r_b_g

def gen_cano_imgs_only(lnb, rad=1, rmr=False):
    gimgs = []
    wi,hi= 300,300

    ptts = lnb.reshape(-1, 3)
    mnxx = np.min(ptts,0)
    mxxx = np.max(ptts,0)
    Lb, wb, hgt, sht = mxxx[1]-mnxx[1], mxxx[0]-mnxx[0], mxxx[-1], mnxx[-1]
    t1 = np.array([Lb,0.0,0.0])
    t3 = np.array([0.0,mnxx[1]-2*max([hgt,abs(sht)]),0])
    t5 = np.array([0.0,0.0,hgt+Lb])
    fc = 0.65
    t2 = np.array([fc*Lb,fc*Lb,-3*max([hgt,abs(sht)])])
    t4 = np.array([-fc*Lb,-fc*Lb,-3*max([hgt,abs(sht)])])
    t6 = np.array([-fc*Lb,fc*Lb,3*max([hgt,abs(sht)])])

    r1 = np.zeros(3)
    r3 = np.array([0,0,-90.0])
    r5 = np.array([0,-90,0])

    target_xyz = np.zeros(3) 
    r2 = get_camera_rotation(t2, target_xyz)
    r4 = get_camera_rotation(t4, target_xyz)
    r6 = get_camera_rotation(t6, target_xyz)

    Ts = [t1,t2,t3,t4,t5,t6] 
    Rs = [r1,r2,r3,r4,r5,r6]


    for pp in range(len(Ts)): #
        t1p = Ts[pp]
        r1p = Rs[pp]
        Rbg = np.array(R.from_euler('xyz',r1p,degrees=True).as_matrix())
        Rgc = np.array(R_c_b.T @ Rbg.T)

        imgi = np.zeros((hi,wi,3)).astype(np.uint8)
        oimg_i = draw_shape(imgi, K_mm_rs,t1p,Rgc,lnb,thickness=rad)
        gimgs.append(oimg_i)
    imgs = np.stack(gimgs, axis=0)
    imgs = torch.from_numpy(imgs).float().permute(0,3,1,2)
    imgs = 2 * (imgs / 255.0) - 1.0
    return gimgs, imgs, Ts, Rs

def compute_uvs(xyz_ins_, Ts, Rs):
    UVs, aviz = [], []
    for pp in range(len(Ts)):
        t1p = Ts[pp]
        r1p = Rs[pp]
        Rbg = np.array(R.from_euler('xyz',r1p,degrees=True).as_matrix())
        Rgc = np.array(R_c_b.T @ Rbg.T)
        uv_rnd, ff = proj_3d2d(xyz_ins_, t1p, Rgc)
        UVs.append(uv_rnd)
        aviz.append(ff.reshape(-1,1))
    vis_mat = np.hstack(aviz)
    return UVs, vis_mat

def data_eval(xyz_in, Ts, Rs, imgs,imgs_f):
    xyz_in = np.array(xyz_in)
    all_inps = []
    ffa = np.array(np.ones_like(xyz_in[:,0]), dtype=bool)
    UVs, vis_mat = compute_uvs(xyz_in, Ts, Rs)
    all_inps = [(imgs,imgs_f), UVs]
    vis_mat_p = torch.tensor(vis_mat, dtype=torch.bool)
    vis_mat_p = vis_mat_p.view(len(vis_mat_p)//9, 9, -1).sum(dim=1)
    vis_mat_p = vis_mat_p > 0
    return all_inps, ffa, vis_mat_p


def pointcloud_metrics(pc1, pc2, threshold):
    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)

    assert pc1.ndim == 2 and pc1.shape[1] == 3
    assert pc2.ndim == 2 and pc2.shape[1] == 3

    # KD trees
    tree_gt = cKDTree(pc2)
    tree_pred = cKDTree(pc1)

    # Reconstruction -> GT
    dist_pred_to_gt, _ = tree_gt.query(pc1, k=1)

    # GT -> Reconstruction
    dist_gt_to_pred, _ = tree_pred.query(pc2, k=1)

    # Precision
    precision = np.mean(dist_pred_to_gt <= threshold)

    # Density
    density = np.sum(dist_pred_to_gt <= threshold)

    # Completeness / Recall
    completeness = np.mean(dist_gt_to_pred <= threshold)

    # F1
    if precision + completeness > 0:
        f1 = 2 * precision * completeness / (precision + completeness)
    else:
        f1 = 0.0

    # Overall accuracy
    # Commonly reported as the mean reconstruction -> GT distance.
    accuracy = np.median(dist_pred_to_gt)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("completeness: ", completeness)
    print('f1 score: ', f1)

    metrics = {
        "precision": precision,
        "completeness": completeness,
        "recall": completeness,
        "f1": f1,
        "accuracy": accuracy,
        "density": density,

        # Useful additional statistics
        "mean_pred_to_gt": np.mean(dist_pred_to_gt),
        "median_pred_to_gt": np.median(dist_pred_to_gt),
        "mean_gt_to_pred": np.mean(dist_gt_to_pred),
        "median_gt_to_pred": np.median(dist_gt_to_pred),

        "pred_to_gt_distances": dist_pred_to_gt,
        "gt_to_pred_distances": dist_gt_to_pred,
    }

    return metrics

