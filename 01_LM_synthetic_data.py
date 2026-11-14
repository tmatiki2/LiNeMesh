"""
LiNeMesh synthetic data generation with Blender (Cycles).

Starting from a user-defined structural line diagram (LD), the script builds randomly
perturbed 3D members around the LD, samples occupancy / SDF points, and renders
field-like images with random camera poses, member textures, ground textures and
sun light. Every generated domain is written in the BridgeNet layout:

    data/synthetic/train/<id>/<id>_render/
        img_<j>.jpg, trans_<j>.npy, rots_<j>.npy   (images and camera poses)
        pts.npy, sdf.npy, line_bars.npy, Km.npy    (samples, SDF labels, LD, intrinsics)

so it can be used directly by 01_train_LiNeMesh.ipynb.

Run it with Blender in background (offline) mode from the repository folder:

    blender -b --factory-startup -P 01_LM_synthetic_data.py -- --n-domains 100 --n-images 50

Options (after the "--"):
    --n-domains N    number of domains to generate (default 100)
    --n-images N     images rendered per domain (default 50)
    --out DIR        output folder (default data/synthetic/train)
    --start-id N     id of the first domain (default: next free id in --out)
    --ld FILE.npy    LD geometry as an (N, 2, 3) array; default: LD_geom() below
    --samples N      Cycles samples per image (default 32)
    --cpu            render on the CPU instead of the GPU
    --seed N         random seed

Optional textures: put member textures (*.png / *.jpg) in assets/textures/ and ground
textures in assets/backgrounds/. When those folders are empty, random colours are used.
"""
import sys
import os
import glob
import argparse
import math
import random
import numpy as np
from mathutils.bvhtree import BVHTree
import bpy
import bmesh
from mathutils import Vector, Euler
import trimesh
from mathutils import Matrix
from mathutils import Quaternion

# ==========================================================
# PATHS AND OPTIONS (all relative to this repository)
# ==========================================================
try:
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    REPO_ROOT = os.getcwd()
TEXTURE_DIR = os.path.join(REPO_ROOT, "assets", "textures")
BACKGROUND_DIR = os.path.join(REPO_ROOT, "assets", "backgrounds")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "synthetic", "train")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(prog="blender -b --factory-startup -P 01_LM_synthetic_data.py --")
ap.add_argument("--n-domains", type=int, default=100)
ap.add_argument("--n-images", type=int, default=50)
ap.add_argument("--out", type=str, default=DEFAULT_OUT)
ap.add_argument("--start-id", type=int, default=None)
ap.add_argument("--ld", type=str, default=None)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--seed", type=int, default=None)
ARGS = ap.parse_args(argv)
if ARGS.seed is not None:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

def _images_in(folder):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(folder, ext))
    return sorted(files)

TEXTURES = _images_in(TEXTURE_DIR)
BACKGROUNDS = _images_in(BACKGROUND_DIR)

#TODO
## --- Define the LD geometry as a Nx2x3 array for N edges and two node per edge----
## An example is given below (or pass your own LD with --ld my_ld.npy)


def LD_geom():
    """
    Generate a generic 3D line-diagram (LD) geometry array.

    Each line element is defined by the coordinates of its two endpoints:
        [[x1, y1, z1],
         [x2, y2, z2]]

    Returns
    -------
    ld_geom : ndarray, shape (N, 2, 3)
        Array of N line elements, each defined by two 3D endpoints.

    Notes
    -----
    The example represents a two-sided truss-like bridge with:
        - Inclined and vertical truss members.
        - Longitudinal top and bottom chords.
        - Transverse deck and end members.

    All dimensions are illustrative and can be modified to define
    other structural configurations.
    """

    # ---------------------------------------------------------
    # Geometric parameters
    # ---------------------------------------------------------
    spacing = 4.5       # Longitudinal panel spacing
    height = 3.0        # Truss height
    width = 2.5         # Transverse bridge width
    support = 1.5    # Deck depth below the bottom chords
    length = 5 * spacing

    # Translate a member to the opposite side of the bridge
    opposite_side = np.array([-width, 0.0, 0.0])

    # ---------------------------------------------------------
    # 1. Inclined truss members (both sides)
    # ---------------------------------------------------------
    truss_members = [
        [[0, 0, 0],       [0, spacing, height]],
        [[0, spacing, height], [0, 2*spacing, 0]],
        [[0, 2*spacing, 0],    [0, 3*spacing, height]],
        [[0, 2*spacing, height], [0, 3*spacing, 0]],
        [[0, 3*spacing, 0],    [0, 4*spacing, height]],
        [[0, 4*spacing, height], [0, 5*spacing, 0]],
    ]

    truss_members = [np.asarray(m) for m in truss_members]
    truss_members += [m + opposite_side for m in truss_members]

    # ---------------------------------------------------------
    # 2. Vertical truss members (both sides)
    # ---------------------------------------------------------
    vertical_members = [
        np.array([[0, spacing, 0], [0, spacing, height]])
        + np.array([0, k * spacing, 0])
        for k in range(4)
    ]

    vertical_members += [m + opposite_side for m in vertical_members]

    # ---------------------------------------------------------
    # 3. Deck and end members
    # ---------------------------------------------------------
    deck_members = [
        np.array([[0, 0, 0], [0, 0, -support]]),
        np.array([[0, length, 0], [0, length, -support]]),
    ]

    deck_members += [m + opposite_side for m in deck_members]

    # ---------------------------------------------------------
    # 4. Longitudinal chords and transverse connections
    # ---------------------------------------------------------
    longitudinal_members = [
        # Top chords
        [[0, spacing, height], [0, length-spacing, height]],
        [[-width, spacing, height], [-width, length-spacing, height]],

        # Bottom chords
        [[0, 0, 0], [0, length, 0]],
        [[-width, 0, 0], [-width, length, 0]],

        # Intermediate longitudinal member
        [[-width/2, 0, 0], [-width/2, length, 0]],

        # Transverse end connections
        [[0, 0, 0], [-width, 0, 0]],
        [[0, length, 0], [-width, length, 0]],
    ]

    longitudinal_members = [
        np.asarray(m) for m in longitudinal_members
    ]

    # ---------------------------------------------------------
    # Assemble all line elements
    # ---------------------------------------------------------
    ld_geom = np.stack(
        truss_members + vertical_members +
        deck_members + longitudinal_members
    )
    return ld_geom

SLD = np.load(ARGS.ld).astype(np.float64) if ARGS.ld else LD_geom() ### subsequently, this LD will be perturbed to generate new examples

def deselect_all():
    for obj in bpy.context.selected_objects:
        obj.select_set(False)

def safe_delete_objects(obj_types=None):
    objs = []
    for obj in list(bpy.data.objects):

        if obj_types is None:
            objs.append(obj)

        elif obj.type in obj_types:
            objs.append(obj)

    for obj in objs:
        bpy.data.objects.remove(obj, do_unlink=True)


def purge_orphans():

    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)

    for block in list(bpy.data.cameras):
        if block.users == 0:
            bpy.data.cameras.remove(block)

    for block in list(bpy.data.lights):
        if block.users == 0:
            bpy.data.lights.remove(block)


def clean_scene():
    safe_delete_objects()
    purge_orphans()


def blender_obj_to_trimesh(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    mesh.calc_loop_triangles()
    verts = np.array(
        [obj_eval.matrix_world @ v.co for v in mesh.vertices],
        dtype=np.float64
    )
    faces = np.array(
        [tri.vertices[:] for tri in mesh.loop_triangles],
        dtype=np.int64
    )
    tm = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False
    )
    obj_eval.to_mesh_clear()
    return tm

def sampts(obj, N=1000, margin=0.5):
    tm = blender_obj_to_trimesh(obj)
    pts_surface, _ = trimesh.sample.sample_surface(tm, count=N)
    mins = tm.bounds[0] - margin
    maxs = tm.bounds[1] + margin
    pts_inside = []
    pts_outside = []
    while len(pts_outside) < N:
        candidates = np.random.uniform(
            low=mins,
            high=maxs,
            size=(max(3 * N, 1000), 3)
        )
        inside_mask = tm.contains(candidates)
        if len(pts_outside) < N:
            pts_outside.extend(candidates[~inside_mask])
    pts_outside = np.asarray(pts_outside[:N])
    points = np.array(pts_outside)
    pts_bounds = np.array(pts_surface)
    return points, pts_bounds

def add_prismatic_member(
        xyz_top,
        xyz_bottom,
        scale_xyz=(1.0,1.0,1.0),
        mesh_type="cube",
        name="member",
        ncuts=20,
        num_points=1000,
        image_path=None,ucol=None):

    dd1 = np.array(xyz_top)
    dd2 = np.array(xyz_bottom)
    dd0 = np.linalg.norm(dd1-dd2)
    if dd0>0:
        xyz_top = Vector(xyz_top)
        xyz_bottom = Vector(xyz_bottom)

        direction = xyz_top - xyz_bottom
        length = direction.length
        if length < 1e-8:
            raise ValueError("Zero-length member.")
        direction_norm = direction.normalized()

        mesh = bpy.data.meshes.new(name + "_mesh")
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        bm = bmesh.new()
        if mesh_type.lower() == "cube":
            bmesh.ops.create_cube(
                bm,
                size=1.0
            )
        elif mesh_type.lower() == "cylinder":
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                segments=24,
                radius1=1.0,
                radius2=1.0,
                depth=1.0
            )
        else:
            raise ValueError("mesh_type must be cube or cylinder")

        bm.to_mesh(mesh)
        bm.free()
        obj.scale = (
            scale_xyz[0],
            scale_xyz[1],
            1.0
        )
        z_axis = Vector((0,0,1))
        rot = z_axis.rotation_difference(direction_norm)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = rot
        midpoint = (xyz_top + xyz_bottom) * 0.5
        obj.location = midpoint
        bpy.context.view_layer.update()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        for v in bm.verts:
            v.co.z *= length
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        sampled_points = sampts(obj, N=5000, margin=0.5)
        has_img = image_path is not None and os.path.exists(image_path)
        if has_img or ucol is not None:
            mat = bpy.data.materials.new(name=name + "_mat")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            out = nodes.new(type='ShaderNodeOutputMaterial')
            out.location = (300, 0)
            links.new(
                bsdf.outputs['BSDF'],
                out.inputs['Surface']
            )
            if has_img and ucol is None:
                image = bpy.data.images.load(image_path)
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.location = (-300, 0)
                tex_node.image = image
                links.new(
                    tex_node.outputs['Color'],
                    bsdf.inputs['Base Color']
                )

            else:
                r = ucol[0] / 255.0
                g = ucol[1] / 255.0
                b = ucol[2] / 255.0
                bsdf.inputs['Base Color'].default_value = (
                    r,
                    g,
                    b,
                    1.0
                )
            obj.data.materials.clear()
            obj.data.materials.append(mat)

        return sampled_points, obj


# ==========================================================
# CAMERA
# ==========================================================
def add_camera_looking_at_target(
        camera_xyz,
        target_xyz,
        K_matrix,
        image_width,
        image_height,
        camera_name="Camera"):

    safe_delete_objects(obj_types={'CAMERA'})
    camera_xyz = Vector(camera_xyz)
    target_xyz = Vector(target_xyz)
    cam_data = bpy.data.cameras.new(camera_name)
    cam_obj = bpy.data.objects.new(camera_name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = camera_xyz
    direction = (target_xyz - camera_xyz).normalized()
    rat = np.random.uniform(0,1)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    x_angle_deg = random.uniform(-15.0, 15.0)
    x_angle_rad = math.radians(x_angle_deg)
    local_x_rot = Quaternion((0, 1, 1), x_angle_rad)
    final_rot = rot_quat @ local_x_rot
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = final_rot
    fx = K_matrix[0,0]
    fy = K_matrix[1,1]
    cx = K_matrix[0,2]
    cy = K_matrix[1,2]
    cam_data.sensor_fit = 'HORIZONTAL'
    sensor_width_mm = 36.0
    cam_data.sensor_width = sensor_width_mm
    f_mm = fx * sensor_width_mm / image_width
    cam_data.lens = f_mm
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy
    cam_data.shift_x = -(cx - image_width / 2.0) / image_width
    cam_data.shift_y = (cy - image_height / 2.0) / image_width
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height
    bpy.context.scene.camera = cam_obj
    bpy.context.view_layer.update()
    R_world2cam = cam_obj.matrix_world.inverted().to_3x3()
    R_world2cam = np.array(R_world2cam)
    R_bb_c = np.array([
        [1,0,0],
        [0,-1,0],
        [0,0,-1]
    ]).astype(np.float32)
    R_world2cam = R_bb_c @ R_world2cam
    return cam_obj, R_world2cam, K_matrix


# ==========================================================
# LIGHTS
# ==========================================================
def add_sun_light_aligned_to_camera(
        camera_obj,
        energy=5.0,
        angle_deg=2.0):
    light_data = bpy.data.lights.new(
        name="Sun",
        type='SUN'
    )
    light_obj = bpy.data.objects.new(
        "Sun",
        light_data
    )
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = camera_obj.location
    light_obj.rotation_mode = camera_obj.rotation_mode
    light_obj.rotation_quaternion = (
        camera_obj.rotation_quaternion.copy()
    )
    light_data.energy = energy
    light_data.angle = math.radians(angle_deg)
    return light_obj


def add_vertical_sun_light(
        location_xyz,
        energy=10.0,
        angle_deg=5.0):

    light_data = bpy.data.lights.new(
        name="VerticalSun",
        type='SUN'
    )

    light_obj = bpy.data.objects.new(
        "VerticalSun",
        light_data
    )

    bpy.context.collection.objects.link(light_obj)

    light_obj.location = Vector(location_xyz)

    light_obj.rotation_mode = 'XYZ'

    light_obj.rotation_euler = Euler((0,0,0), 'XYZ')

    light_data.energy = energy

    light_data.angle = math.radians(angle_deg)

    return light_obj


# ==========================================================
# ENVIRONMENT
# ==========================================================
def set_environment_background(
        image_path,
        strength=1.0):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    env_node = nodes.new(type="ShaderNodeTexEnvironment")
    bg_node = nodes.new(type="ShaderNodeBackground")
    out_node = nodes.new(type="ShaderNodeOutputWorld")
    if os.path.exists(image_path):
        env_image = bpy.data.images.load(image_path)
        env_node.image = env_image
    bg_node.inputs["Strength"].default_value = strength
    links.new(
        env_node.outputs["Color"],
        bg_node.inputs["Color"]
    )
    links.new(
        bg_node.outputs["Background"],
        out_node.inputs["Surface"]
    )


# ==========================================================
# RENDER
# ==========================================================
def setup_gpu(use_gpu=True):
    """Use the first available GPU backend (OptiX, CUDA, HIP, Metal, oneAPI); fall back to the CPU."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    if use_gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                try:
                    prefs.compute_device_type = backend
                except TypeError:
                    continue
                if hasattr(prefs, "refresh_devices"):
                    prefs.refresh_devices()
                else:
                    prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == backend]
                if gpus:
                    for d in prefs.devices:
                        d.use = (d.type == backend)
                    scene.cycles.device = 'GPU'
                    print("Cycles device:", backend, [d.name for d in gpus])
                    return backend
        except Exception as e:
            print("GPU setup failed, using CPU:", e)
    scene.cycles.device = 'CPU'
    print("Cycles device: CPU")
    return "CPU"


def render_with_cycles(
        output_path,
        image_width=640,
        image_height=480,
        samples=32): #64

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = image_width
    scene.render.resolution_y = image_height
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    return output_path



def insert_bay(SLD_, scale_xyzs, image_path1=None, b_type="cylinder", ucol=None):
    spts_ins = []
    spts_ots = []
    for brs in SLD_:
        tb = tuple(brs[0].tolist())
        bb = tuple(brs[1].tolist())
        ou1 = add_prismatic_member(tb,bb,scale_xyzs,mesh_type=b_type,image_path=image_path1,ucol=ucol)
        if ou1 is not None:
            pts_o, pts = ou1[0]
            spts_ins.append(pts)
            spts_ots.append(pts_o)

    apts_in = np.vstack(spts_ins)
    apts_ot = np.vstack(spts_ots)
    return apts_in, apts_ot

def add_ground_plane(
        z=0.0,
        size=100.0,
        image_path=None,
        name="GroundPlane",
        color=None):
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(0.0, 0.0, z)
    )
    plane = bpy.context.active_object
    plane.name = name
    if image_path is not None:
        mat = bpy.data.materials.new(name + "_Material")
        mat.use_nodes = True

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        tex_node = nodes.new("ShaderNodeTexImage")
        bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
        out_node = nodes.new("ShaderNodeOutputMaterial")
        tex_node.location = (-400, 0)
        bsdf_node.location = (-100, 0)
        out_node.location = (200, 0)
        img = bpy.data.images.load(image_path)
        tex_node.image = img
        links.new(tex_node.outputs["Color"],
                  bsdf_node.inputs["Base Color"])

        links.new(bsdf_node.outputs["BSDF"],
                  out_node.inputs["Surface"])
        if plane.data.materials:
            plane.data.materials[0] = mat
        else:
            plane.data.materials.append(mat)

        bpy.context.view_layer.objects.active = plane
        plane.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(method='ANGLE_BASED')
        bpy.ops.object.mode_set(mode='OBJECT')
    elif color is not None:
        mat = bpy.data.materials.new(name + "_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        plane.data.materials.append(mat)
    bpy.context.view_layer.update()
    return plane


from scipy.spatial import cKDTree 
def knn(pcd1,pcd2,th=0.01):
    tree = cKDTree(pcd2)
    dists, idx = tree.query(pcd1, k=1)
    dists = -1.0*np.array(dists).flatten()
    return dists

def replace_nans(arr):
    return np.nan_to_num(arr, nan=0.0)  
    

# ==========================================================
# DATASET GENERATION
# ==========================================================
K_mm = np.array([
    [385.477716,0.0,320],
    [0.0,385.477716,240],
    [0.0,0.0,1.0]
])

K_mm_r = np.array([
    [385.464,0,330.348],
    [0,385.093,241.54],
    [0,0,1]
])

K_mm_od = np.array([
    [508.3997,0,316.0652],
    [0,677.8663,254.0068],
    [0,0,1]
])

Kd_3803 = np.array([[      11597,           0,        4171],
       [          0,       11364,      5529.6],
       [          0,           0,           1]])
sx,sy = 640/8736, 480/11648 ##original
K_dls = np.array(Kd_3803)
K_dls[0,0] = sx*K_dls[0,0]
K_dls[0,2] = sx*K_dls[0,2]
K_dls[1,1] = sy*K_dls[1,1]
K_dls[1,2] = sy*K_dls[1,2]
Ks = [K_mm, K_mm_r, K_dls]


# ==========================================================
# MAIN LOOP
# ==========================================================
Ndoms = ARGS.n_domains   ### number of domains (bridges) to generate
Nimgs = ARGS.n_images    ### number of images to render per domain
save_dir = ARGS.out      ### output folder (BridgeNet layout)
os.makedirs(save_dir, exist_ok=True)
existing = [int(d) for d in os.listdir(save_dir) if d.isdigit()]
start_id = ARGS.start_id if ARGS.start_id is not None else (max(existing) + 1 if existing else 0)
print("Output folder:", save_dir, "| first domain id:", start_id)
print("Member textures:", len(TEXTURES), "| ground textures:", len(BACKGROUNDS), "(random colours when 0)")

kdoms = 0
while kdoms < Ndoms:
        print("current dom:", kdoms)
        clean_scene()
        setup_gpu(use_gpu=not ARGS.cpu)
        ## --- generate random global perturbation to LD --- other forms of anisotropic perturbations can be applied
        rng = 1.0 ## 2 meters perturbation --- reset as needed
        dxdydz = np.random.uniform(-rng,rng,3)
        SLD_ = SLD + dxdydz.reshape(1,1,-1)

        ldpts = SLD_.reshape(-1,3)
        ctr, maxx = np.min(ldpts,0), np.max(ldpts,0)
        SLD_ = SLD_ - ctr.reshape(1,1,3) ## LD shifting

        maxx = maxx - ctr
        lb, hgt, wb = maxx[1], maxx[2], maxx[0]
        sx = np.random.uniform(0.01,0.1) # --- set the ranges of member dimensions
        scl_xyz=tuple([sx]*3)
        b_type = random.choice(["cylinder", "cube"])

        if TEXTURES:
            image_path, ucol = random.choice(TEXTURES), None
        else:
            image_path, ucol = None, tuple(np.random.randint(40, 230, 3).tolist())
        pcd2, pcd1 = insert_bay(SLD_, scl_xyz, image_path, b_type, ucol=ucol)

        pcd1 = replace_nans(pcd1)
        pcd2 = replace_nans(pcd2)
        pcd1 = np.ascontiguousarray(pcd1, dtype=np.float64)
        pcd2 = np.ascontiguousarray(pcd2, dtype=np.float64)

        sdfs = knn(pcd1,pcd2,th=0.001)
        pts = np.vstack((pcd1,pcd2))
        sdfs = np.hstack((sdfs,np.zeros(len(pcd2))))

        gp_imgp = random.choice(BACKGROUNDS) if BACKGROUNDS else None
        gp_col = tuple(np.random.uniform(0.15, 0.6, 3).tolist())
        add_ground_plane(z=0.0,size=500.0,image_path=gp_imgp,name="GroundPlane",color=gp_col)

        bridge_id = f"{start_id + kdoms:05d}"
        save_dirn = os.path.join(save_dir, bridge_id, bridge_id + "_render") + os.sep
        if not(os.path.exists(save_dirn)):
            os.makedirs(save_dirn)

        Km = random.choice(Ks)
        np.save(save_dirn + 'pts.npy', pts.astype(np.float16))
        np.save(save_dirn + 'sdf.npy', sdfs.astype(np.float16))
        np.save(save_dirn + 'line_bars.npy', SLD_.astype(np.float16))
        np.save(save_dirn + 'Km.npy', Km.astype(np.float16))

        ntrk = random.sample(range(len(pts)),Nimgs)
        eng2 = np.random.uniform(0.5,5)
        sunx = np.random.uniform(-3,wb+3)
        suny = np.random.uniform(-5.0,lb+5)
        sunz = np.random.uniform(10*hgt,20*hgt)
        sun_loc = (sunx, suny, sunz)
        add_vertical_sun_light(sun_loc,energy=eng2)
        
        ww, hh = 640, 480
        for jj in range(Nimgs):
            xc, yc, zc = np.random.uniform(wb+1.0,min([12.0,lb])), np.random.uniform(-3.0,lb+3.0), np.random.uniform(0,2.5*hgt)
            cxyz = (xc,yc,zc)
            txyz = tuple(pts[ntrk[jj]].tolist())
            camera_obj, R_world2cam, Kbb = add_camera_looking_at_target(cxyz,txyz,Km,ww,hh)
            red_dir = save_dirn + 'img_' + str(jj)
            render_with_cycles(red_dir, image_width=ww, image_height=hh, samples=ARGS.samples)
            trans = np.array(cxyz).flatten()
            np.save(save_dirn + 'trans_' + str(jj) + '.npy', trans.astype(np.float16))
            np.save(save_dirn + 'rots_' + str(jj) + '.npy', R_world2cam.astype(np.float16))
        kdoms += 1
print("DATASET GENERATION COMPLETE")

# Run from the repository folder (see README):
#   blender -b --factory-startup -P 01_LM_synthetic_data.py -- --n-domains 100 --n-images 50