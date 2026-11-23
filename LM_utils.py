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


# ============================================================================ FE mesh assembly
# Vertex order of a hexahedron (VTK_HEXAHEDRON = 12, Abaqus C3D8, Gmsh type 5):
# bottom face (z-) counter-clockwise seen from +z, then the top face (z+).
HEX_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.int64)
# the six faces of a hexahedron in the vertex order above (outward normals)
HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)


def _merge_close(points, tol):
    """Indices merging points closer than tol: rounding to a tol grid, then a KD-tree pass
    that also merges coincident points split by a rounding boundary."""
    keys = np.round(points / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    reps = points[first]
    pairs = cKDTree(reps).query_pairs(tol, output_type="ndarray")
    if len(pairs):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        n = len(reps)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        _, lab = np.unique(lab, return_inverse=True)
        inverse = lab.reshape(-1)[inverse]
    return inverse


def points_to_fe_mesh(points, tol=1e-4, element_size=None, return_info=False):
    """
    Convert the (N*9, 3) LiNeMesh output into an FE mesh: unique nodes and 8-node
    hexahedral connectivity.

    Each group of 9 points is one element: [0] the centroid, [1:9] the 8 corners
    (in any order; they are re-ordered to the VTK / Abaqus C3D8 convention).

    Parameters
    ----------
    points : (N*9, 3) array
    tol : float
        Nodes closer than tol are merged into one node (their mean coordinate).
    element_size : None, float, (3,) array or "auto"
        None: the corners are used as given.
        float / (3,): each element is rebuilt as a box of this edge length about its
        centroid, so that neighbouring elements of a regular coarse mesh share faces
        (a conforming mesh) even when the candidate cubes used at inference overlap.
        "auto": the coarse-mesh spacing is inferred per axis from the centroids, which are
        snapped to that lattice (useful for saved outputs stored in reduced precision).
    return_info : bool
        Also return a dictionary of mesh diagnostics.

    Returns
    -------
    nodes : (M, 3) float64       unique global nodes
    elements : (K, 8) int64      connectivity (global node ids), K <= N
    keep : (K,) int64            index of each returned element in the input
    info : dict (optional)       counts of merged nodes, removed elements, connected
                                 components, overlapping elements and boundary faces
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] % 9:
        raise ValueError("points must be an (N*9, 3) array")
    n_el = points.shape[0] // 9
    grp = points.reshape(n_el, 9, 3)
    cen = grp[:, 0, :]

    if isinstance(element_size, str):
        if element_size != "auto":
            raise ValueError('element_size must be None, a number, a (3,) array or "auto"')
        h = np.array([np.median(np.diff(np.unique(cen[:, i]))) if len(np.unique(cen[:, i])) > 1 else 1.0
                      for i in range(3)])
        o = cen.min(0)
        cen = o + np.round((cen - o) / h) * h
        element_size = h
    if element_size is not None:
        half = 0.5 * np.broadcast_to(np.asarray(element_size, dtype=np.float64), (3,))
        corners = cen[:, None, :] + HEX_SIGNS[None, :, :] * half[None, None, :]
    else:
        corners = grp[:, 1:, :]
        # re-order the corners to the VTK convention from their octant about the centroid
        sg = np.sign(corners - cen[:, None, :]).astype(np.int64)
        if np.any(sg == 0):
            raise ValueError("degenerate element: a corner lies on a centroid plane")
        code = (sg[..., 0] > 0) * 1 + (sg[..., 1] > 0) * 2 + (sg[..., 2] > 0) * 4
        ref = (HEX_SIGNS[:, 0] > 0) * 1 + (HEX_SIGNS[:, 1] > 0) * 2 + (HEX_SIGNS[:, 2] > 0) * 4
        lut = np.full(8, -1); lut[ref] = np.arange(8)
        pos = lut[code]
        if np.any(np.sort(pos, axis=1) != np.arange(8)):
            raise ValueError("an element does not have one corner per octant (not an axis-aligned hexahedron)")
        ordered = np.empty_like(corners)
        ordered[np.arange(n_el)[:, None], pos] = corners
        corners = ordered

    flat = corners.reshape(-1, 3)
    inverse = _merge_close(flat, tol)
    n_nodes = int(inverse.max()) + 1
    nodes = np.zeros((n_nodes, 3))
    np.add.at(nodes, inverse, flat)
    nodes /= np.bincount(inverse, minlength=n_nodes)[:, None]
    elements = inverse.reshape(n_el, 8).astype(np.int64)

    # remove degenerate (collapsed) and duplicated elements
    srt = np.sort(elements, axis=1)
    degenerate = np.any(srt[:, 1:] == srt[:, :-1], axis=1)
    _, first = np.unique(srt, axis=0, return_index=True)
    unique_mask = np.zeros(n_el, bool); unique_mask[first] = True
    keep = np.where(unique_mask & ~degenerate)[0]
    elements = elements[keep]

    # drop nodes no longer referenced and renumber
    used = np.unique(elements)
    remap = -np.ones(n_nodes, np.int64); remap[used] = np.arange(len(used))
    nodes, elements = nodes[used], remap[elements]

    if not return_info:
        return nodes, elements, keep
    faces, owner, count = hex_faces(elements)
    info = dict(elements_in=n_el, elements_out=len(elements), nodes=len(nodes),
                corner_points_in=len(flat), degenerate_removed=int(degenerate.sum()),
                duplicates_removed=int(n_el - unique_mask.sum()),
                boundary_faces=int(np.sum(count == 1)),
                connected_components=int(mesh_components(elements)),
                overlapping_elements=int(count_overlaps(nodes, elements)))
    return nodes, elements, keep, info


def hex_faces(elements):
    """Unique faces of a hexahedral mesh: (F, 4) node ids, an owner element per face and
    the number of elements sharing the face (1 = boundary face, 2 = interior face)."""
    quads = elements[:, HEX_FACES].reshape(-1, 4)
    owner = np.repeat(np.arange(len(elements)), 6)
    key = np.sort(quads, axis=1)
    _, first, inv, count = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    return quads[first], owner[first], count


def mesh_components(elements):
    """Number of face-connected components of a hexahedral mesh."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(elements) == 0:
        return 0
    key = np.sort(elements[:, HEX_FACES].reshape(-1, 4), axis=1)
    owner = np.repeat(np.arange(len(elements)), 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    a, b = order[:-1], order[1:]
    same = inv[a] == inv[b]
    rows, cols = owner[a[same]], owner[b[same]]
    n = len(elements)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return connected_components(g, directed=False)[0]


def count_overlaps(nodes, elements):
    """Number of elements whose interior contains the centroid of another element
    (0 for a conforming mesh of the coarse grid)."""
    if len(elements) < 2:
        return 0
    xyz = nodes[elements]
    lo, hi = xyz.min(1), xyz.max(1)
    cen = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    tree = cKDTree(cen)
    pairs = tree.query_pairs(float(np.max(half)) * 2.0, output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    gap = np.abs(cen[i] - cen[j]) - (half[i] + half[j])
    over = np.all(gap < -1e-9 * (1 + np.abs(cen[i]).max()), axis=1)
    return int(len(np.unique(np.concatenate([i[over], j[over]]))))


def save_fe_mesh(path, nodes, elements, cell_data=None):
    """Write the hexahedral mesh to a legacy VTK (.vtk, ParaView) or Abaqus (.inp) file.
    Nodes and elements are numbered from 1 in the .inp file. Materials, sections, loads and
    boundary conditions are not written; they are defined in the FE software."""
    path = str(path)
    if path.endswith(".vtk"):
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\nLiNeMesh hexahedral mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {len(nodes)} double\n")
            np.savetxt(f, nodes, fmt="%.6f")
            f.write(f"CELLS {len(elements)} {9 * len(elements)}\n")
            np.savetxt(f, np.hstack([np.full((len(elements), 1), 8), elements]), fmt="%d")
            f.write(f"CELL_TYPES {len(elements)}\n")
            np.savetxt(f, np.full(len(elements), 12), fmt="%d")
            if cell_data:
                f.write(f"CELL_DATA {len(elements)}\n")
                for name, val in cell_data.items():
                    f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                    np.savetxt(f, np.asarray(val, dtype=np.float64).reshape(-1), fmt="%.6f")
    elif path.endswith(".inp"):
        with open(path, "w") as f:
            f.write("*HEADING\nLiNeMesh hexahedral mesh (C3D8)\n*NODE, NSET=NALL\n")
            np.savetxt(f, np.hstack([np.arange(1, len(nodes) + 1)[:, None], nodes]), fmt=["%d", "%.6f", "%.6f", "%.6f"], delimiter=", ")
            f.write("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
            np.savetxt(f, np.hstack([np.arange(1, len(elements) + 1)[:, None], elements + 1]), fmt="%d", delimiter=", ")
    else:
        raise ValueError("use a .vtk or .inp file name")


def unique_points(points, tol=1e-4):
    """Points of the (N*9, 3) output with coincident points (closer than tol) merged;
    these are the points evaluated by pointcloud_metrics in the paper."""
    points = np.asarray(points, dtype=np.float64)
    inverse = _merge_close(points, tol)
    _, first = np.unique(inverse, return_index=True)
    return points[np.sort(first)]


# ============================================================================ metrics
def voxel_iou(pc1, pc2, voxel, margin=0.5, fill=True):
    """
    Volumetric IoU of a reconstruction pc1 and a reference pc2 (paper protocol).
    Both point sets are voxelised on a common grid of edge `voxel`, anchored at the
    reference bounding box enlarged by `margin`; a voxel is occupied if it contains a point,
    enclosed interiors are filled, and IoU = |A and B| / |A or B|. Points of pc1 outside the
    enlarged box are counted as occupied voxels (without filling).
    Returns IoU and the occupied volumes of pc1 and pc2 (number of voxels).
    """
    from scipy import ndimage
    lo = pc2.min(0) - margin
    hi = pc2.max(0) + margin
    shape = np.floor((hi - lo) / voxel).astype(np.int64) + 1

    def occ(X):
        k = np.floor((X - lo) / voxel).astype(np.int64)
        inside = np.all((k >= 0) & (k < shape), axis=1)
        g = np.zeros(tuple(shape), bool)
        g[tuple(k[inside].T)] = True
        if fill:
            g = ndimage.binary_fill_holes(g)
        outside = len(np.unique(k[~inside], axis=0)) if (~inside).any() else 0
        return g, outside

    a, ao = occ(np.asarray(pc1, dtype=np.float64))
    b, bo = occ(np.asarray(pc2, dtype=np.float64))
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + ao + bo
    return float(inter / union) if union else 0.0, int(a.sum() + ao), int(b.sum() + bo)


def pointcloud_metrics(pc1, pc2, threshold, iou=True, verbose=True):
    """
    Metrics of a reconstruction pc1 against the reference pc2 at distance threshold tau,
    as defined in the paper:
      precision  P   fraction of pc1 within tau of pc2
      recall     R   fraction of pc2 within tau of pc1 (completeness)
      f1             2PR / (P + R)
      accuracy   A   MEAN distance from pc1 to pc2 (same units as the points)
      iou            volumetric IoU on voxels of edge tau (enclosed interiors filled)
      vol_ratio      occupied volume of pc1 / occupied volume of pc2 (1 is ideal)
    For LiNeMesh, pass the unique mesh points (see unique_points).
    """
    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)
    assert pc1.ndim == 2 and pc1.shape[1] == 3
    assert pc2.ndim == 2 and pc2.shape[1] == 3

    dist_pred_to_gt, _ = cKDTree(pc2).query(pc1, k=1)
    dist_gt_to_pred, _ = cKDTree(pc1).query(pc2, k=1)

    precision = float(np.mean(dist_pred_to_gt <= threshold))
    completeness = float(np.mean(dist_gt_to_pred <= threshold))
    f1 = 2 * precision * completeness / (precision + completeness) if precision + completeness > 0 else 0.0
    accuracy = float(np.mean(dist_pred_to_gt))

    metrics = {
        "precision": precision,
        "completeness": completeness,
        "recall": completeness,
        "f1": f1,
        "accuracy": accuracy,
        "mean_pred_to_gt": accuracy,
        "median_pred_to_gt": float(np.median(dist_pred_to_gt)),
        "mean_gt_to_pred": float(np.mean(dist_gt_to_pred)),
        "median_gt_to_pred": float(np.median(dist_gt_to_pred)),
        "pred_to_gt_distances": dist_pred_to_gt,
        "gt_to_pred_distances": dist_gt_to_pred,
    }
    if iou:
        v, va, vb = voxel_iou(pc1, pc2, threshold)
        metrics.update(iou=v, vol_ratio=va / vb if vb else float("nan"))
    if verbose:
        print(f"precision: {precision:.3f}  recall: {completeness:.3f}  f1: {f1:.3f}  "
              f"mean accuracy: {accuracy:.4f}" + (f"  IoU: {metrics['iou']:.3f}  V/Vref: {metrics['vol_ratio']:.2f}" if iou else ""))
    return metrics
