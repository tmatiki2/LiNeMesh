# LiNeMesh

**Analysis mesh generation from images and structural line diagrams**

LiNeMesh generates the analysis mesh (the discretized analysis domain) of civil structures for finite element analysis (FEA) from multi-view images (UAV or handheld photos, or renders of a CAD/BIM model) and a structural line diagram (LD). It classifies the elements of a user-prescribed coarse hexahedral mesh as occupied or empty. The LD provides the spatial reference, so no per-image camera pose estimation is needed, and the element size is set by the user. The mesh is metric when the LD is metric (from CAD/BIM or a few field measurements); with a schematic LD it is defined in the LD frame. Materials, boundary conditions and loads are defined afterwards in the FE software.

![LiNeMesh graphical abstract](assets/graphical_abstract.png)

---

## Contents

- [Repository structure](#repository-structure)
- [1. Installation](#1-installation)
- [2. Download the BridgeNet dataset](#2-download-the-bridgenet-dataset)
- [3. Download the pretrained model](#3-download-the-pretrained-model)
- [4. Evaluation: generate a mesh with the pretrained model](#4-evaluation-generate-a-mesh-with-the-pretrained-model)
- [5. Training](#5-training)
- [6. Synthetic data from your own line diagram (Blender)](#6-synthetic-data-from-your-own-line-diagram-blender)
- [Data format](#data-format)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## Repository structure

```
LiNeMesh/
├── 01_train_LiNeMesh.ipynb       # training notebook
├── 01_LM_mesh_viz.py             # evaluation: mesh generation, FE mesh export, metrics, visualisation
├── 01_LM_synthetic_data.py       # synthetic data generation from a user-defined LD (Blender)
├── LM_models.py                  # LiNeMesh network
├── LM_utils.py                   # canonical cameras, projection, FE mesh assembly/export, metrics
├── LM_paths.py                   # all paths and download locations (relative to the repo)
├── requirements.txt
├── scripts/
│   ├── download_bridgenet.py     # downloads the release parts, rebuilds/extracts BridgeNet into data/bridgenet/
│   ├── download_weights.py       # downloads the pretrained model into weights/
│   ├── mesh_from_points.py       # saved (N*9, 3) output -> FE mesh (.vtk/.inp) and metrics
│   ├── reproduce_table2.py       # LiNeMesh rows of Table II on the five test bridges
│   └── install_blender_deps.py   # installs trimesh/scipy/rtree into Blender's Python
├── data/
│   ├── bridgenet_manifest.json   # release parts of the dataset (names, sizes, SHA-256)
│   └── bridgenet_test_ids.json   # test ids of the paper and test domains with ground truth
├── weights/                      # pretrained / trained models (*.pth, downloaded)
├── assets/
│   ├── graphical_abstract.png
│   ├── textures/                 # optional member textures for synthetic data
│   └── backgrounds/              # optional ground textures for synthetic data
└── outputs/                      # generated meshes (created on first run)
```

Every path is resolved relative to the cloned repository in `LM_paths.py`, so the scripts and the notebook run as they are once the repository is cloned. To keep the dataset on another disk, set the environment variable `LINEMESH_DATA` to that folder.

---

## 1. Installation

**Clone the repository**

```bash
git clone https://github.com/tmatiki2/LiNeMesh.git
cd LiNeMesh
```

**Create an environment** (Python 3.10 or 3.11 recommended)

With conda:

```bash
conda create -n linemesh python=3.10 -y
conda activate linemesh
```

or with venv:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Install the dependencies**

For GPU training, first install the CUDA build of PyTorch for your system from [pytorch.org](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the requirements:

```bash
pip install -r requirements.txt
```

The ImageNet weights of the ResNet backbones are downloaded automatically by `torchvision` the first time the model is built.

---

## 2. Download the BridgeNet dataset

BridgeNet (synthetic) holds **5000 training domains** (ids `00000`–`04999`) and **500 test domains** (ids `05000`–`05499`). It is published on the [v1.0 release](https://github.com/tmatiki2/LiNeMesh/releases/tag/v1.0) as one ZIP archive (22.3 GB) split into twelve binary parts, `part_000` … `part_011`, together with `original.sha256`, the SHA-256 checksum of the complete archive.

**With the script (recommended):**

```bash
python scripts/download_bridgenet.py --split test   # test split only: downloads 3 parts (~4.3 GB)
python scripts/download_bridgenet.py --split train  # training split only
python scripts/download_bridgenet.py                # everything (~22.3 GB download)
```

- The full download concatenates the parts in order (the same as `cat`), verifies the SHA-256 checksum and extracts `train/` and `test/`. Allow about 95 GB of free disk space during this step (parts, archive and ~48 GB of extracted data). Use `--keep-parts` or `--keep-zip` to keep the intermediate files.
- `--split test` or `--split train` downloads only the parts that hold that split and extracts it directly from the parts, without writing the 22 GB archive. Every file is checked against its CRC-32.
- Interrupted downloads are resumed when the script is run again.
- If you downloaded the parts yourself, pass their folder with `--parts-dir DIR`.

**Manually.** Download all twelve parts and `original.sha256` from the release into one folder, then:

```bash
# Linux / macOS
cat part_000 part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008 part_009 part_010 part_011 > bridgenet.zip
sha256sum bridgenet.zip            # macOS: shasum -a 256 bridgenet.zip  (must match original.sha256)
unzip bridgenet.zip -d data/       # creates data/bridgenet/train and data/bridgenet/test
```

```bat
:: Windows (Command Prompt)
copy /b part_000+part_001+part_002+part_003+part_004+part_005+part_006+part_007+part_008+part_009+part_010+part_011 bridgenet.zip
certutil -hashfile bridgenet.zip SHA256
```

On Windows, extract `bridgenet.zip` into `data\` with 7-Zip or `tar -xf bridgenet.zip -C data`.

The result is:

```
data/bridgenet/train/<id>/<id>_render/    5000 training domains
data/bridgenet/test/<id>/<id>_render/      500 test domains
```

To keep the dataset on another disk, set `LINEMESH_DATA` to the folder that should contain `train/` and `test/`.

The five test bridges reported in Table II of the paper (05015, 05209, 05311, 05328, 05384) and the 31 test domains that include a ground-truth point cloud (`dense_pts.npy`) are listed in `data/bridgenet_test_ids.json`.

---

## 3. Download the pretrained model

```bash
python scripts/download_weights.py
```

This saves `weights/mod_brdgnet_ptrain.pth` (the BridgeNet model of the paper) from the v1.0 release. You can skip this step, because `01_LM_mesh_viz.py` downloads the weights automatically on its first run.

---

## 4. Evaluation: generate a mesh with the pretrained model

```bash
python 01_LM_mesh_viz.py
```

The script:
1. picks a BridgeNet test domain (by default one of the five of Table II: 05015, 05209, 05311, 05328, 05384) and selects 25 of its field images with a fixed seed;
2. builds a coarse hexahedral mesh around the LD with a prescribed element size;
3. renders the canonical LD views and combines them with the field images;
4. keeps the elements whose predicted occupancy is at least 50 %, with their occupancy confidence;
5. assembles the FE mesh: unique nodes (coincident nodes closer than 10⁻⁴ m merged) and 8-node hexahedral connectivity, and reports its number of nodes, elements, connected components and boundary faces;
6. reports the metrics of the paper against the ground-truth point cloud at τ = 0.05 and τ = 0.10 m;
7. saves the mesh to `outputs/` and shows its boundary faces coloured by occupancy confidence in an Open3D window.

**Metrics** (computed on the unique mesh points, as in the paper):

| Metric | Definition |
|---|---|
| Precision *P* | fraction of mesh points within τ of the reference |
| Recall *R* | fraction of reference points within τ of the mesh (completeness) |
| *F*1 | 2*PR* / (*P* + *R*) |
| Accuracy *A* | mean distance from the mesh points to the reference |
| IoU | volumetric intersection-over-union of the occupied voxels of edge τ (enclosed interiors filled) |
| *V*/*V*ref | ratio of the occupied volumes (1 is ideal; > 1 excess, < 1 missing material) |

**Outputs** (`outputs/`):

| File | Content |
|---|---|
| `linemesh_<id>.vtk` | hexahedral mesh with the occupancy confidence of each element (open in ParaView) |
| `linemesh_<id>.inp` | the same mesh as Abaqus C3D8 elements (nodes and elements only) |
| `linemesh_<id>_nodes.npy`, `linemesh_<id>_elements.npy` | unique nodes `(M, 3)` and connectivity `(K, 8)` (VTK / C3D8 vertex order) |
| `linemesh_<id>_mesh_points.npy` | raw output `(N*9, 3)`: centroid and 8 corners of each occupied element |
| `linemesh_<id>_metrics.json` | metrics, mesh statistics (nodes, elements, connected components) and the settings of the run |

The candidate cubes evaluated by the network may overlap; the FE mesh rebuilds each occupied element as a cube of the coarse-mesh size about its centroid, so neighbouring elements share faces (a conforming mesh).

Options:

| Option | Default | Description |
|---|---|---|
| `--sample ID` | random paper sample | test domain id, e.g. `--sample 5040` |
| `--n-views N` | 25 | number of field images used |
| `--mesh-size H` | 0.05 | element size of the coarse mesh |
| `--weights FILE` | `weights/mod_brdgnet_ptrain.pth` | model weights |
| `--no-vis` | off | do not open a window (servers / SSH); results are still saved |
| `--seed N` | 0 | random seed for the selection of the field images |
| `--wireframe` | off | also draw the element edges |

Example:

```bash
python 01_LM_mesh_viz.py --sample 5015 --mesh-size 0.05 --no-vis
```

**Reproducing Table II (LiNeMesh rows).**

```bash
python scripts/download_bridgenet.py --split test
python scripts/reproduce_table2.py
```

This runs the pretrained model on the five test bridges of the paper and writes the table to `outputs/table2_linemesh.md`. The selection of field images is random (fixed seed), so the metrics may differ slightly from the paper; use `--seed` to try other selections. The competing methods of Table II (RealityCapture, COLMAP, Gaussian Splatting, MUSt3R, NeuS) were run with their public implementations and are not part of this repository.

**Converting a saved output.** Any `(N*9, 3)` LiNeMesh output can be converted to an FE mesh, and evaluated against a reference point cloud, with:

```bash
python scripts/mesh_from_points.py outputs/linemesh_05015_mesh_points.npy --ref reference.npy --tau 0.05 0.10
```

or from Python:

```python
from LM_utils import points_to_fe_mesh, save_fe_mesh, unique_points, pointcloud_metrics
nodes, elements, keep, info = points_to_fe_mesh(points, tol=1e-4, element_size="auto", return_info=True)
save_fe_mesh("mesh.vtk", nodes, elements)                      # or "mesh.inp"
m = pointcloud_metrics(unique_points(points), reference, threshold=0.05)
```

---

## 5. Training

```bash
jupyter notebook 01_train_LiNeMesh.ipynb
```

Run all cells. The notebook:
- reads the training domains from `data/bridgenet/train` (no paths to edit);
- trains the network with the settings used in the paper;
- saves the best checkpoint to `weights/mod_lm_brdnet.pth`.

To train on your own synthetic domains (Section 6), set `DATA_DIR = SYNTH_DIR / "train"` in the data-loading cell. To evaluate your trained model, run:

```bash
python 01_LM_mesh_viz.py --weights weights/mod_lm_brdnet.pth
```

A CUDA GPU is strongly recommended for training.

---

## 6. Synthetic data from your own line diagram (Blender)

`01_LM_synthetic_data.py` generates training domains for a user-defined LD (Section 3.4 of the paper). For each domain it:
- scales the LD by random factors along X, Y and Z and displaces every LD node randomly, keeping the members connected (same topology, different dimensions);
- builds a prismatic or cylindrical member around every LD bar, each with its own random width and thickness;
- samples query points outside, on and inside the members and stores their signed distance to the member surfaces (negative outside, 0 on, positive inside), testing containment against all members; the training notebook labels points with a signed distance ≥ −1 mm as occupied;
- renders field-like images with random cameras, intrinsics, textures and lighting in Blender Cycles.

BridgeNet itself was generated with the parametric bridge generator of the BridgeNet dataset; this script is the generator used for the existing structures of the paper, with their schematic LDs and ranges (Appendix A1).

The output uses the BridgeNet layout, so the training notebook can use it directly.

**6.1 Install Blender** (4.2 or newer; developed with Blender 5.1) from [blender.org](https://www.blender.org/download/), and make sure the `blender` command is available. Otherwise, use the full path to the executable in the commands below, e.g. `~/blender-5.1.2-linux-x64/blender` or `"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"`.

**6.2 Install the extra packages into Blender's Python** (once):

```bash
blender -b --factory-startup -P scripts/install_blender_deps.py
```

**6.3 Define your line diagram.** Edit `LD_geom()` near the top of `01_LM_synthetic_data.py`. It returns an `(N, 2, 3)` array with the two end nodes `[x, y, z]` of each of the N members; the provided example is a two-sided truss bridge. Alternatively, save your LD as a `.npy` file and pass it with `--ld`.

**6.4 Generate the data** in Blender's background (offline) mode, from the repository folder:

```bash
blender -b --factory-startup -P 01_LM_synthetic_data.py -- --n-domains 100 --n-images 50
```

With your own LD file:

```bash
blender -b --factory-startup -P 01_LM_synthetic_data.py -- --ld my_ld.npy --n-domains 500
```

Everything after `--` is passed to the script:

| Option | Default | Description |
|---|---|---|
| `--n-domains N` | 100 | number of domains to generate |
| `--n-images N` | 50 | images rendered per domain |
| `--out DIR` | `data/synthetic/train` | output folder |
| `--start-id N` | next free id | id of the first domain (useful for several parallel runs) |
| `--ld FILE` | `LD_geom()` | LD geometry as an `(N, 2, 3)` `.npy` array |
| `--samples N` | 32 | Cycles samples per image |
| `--cpu` | off | render on the CPU (by default the first available GPU is used: OptiX, CUDA, HIP, Metal or oneAPI) |
| `--seed N` | none | random seed |
| `--scale-range A B` | 0.8 1.2 | range of the random global scale factors along X, Y, Z |
| `--node-jitter D` | 0.05 | maximum random displacement of each LD node along X, Y, Z (m) |
| `--member-size A B` | 0.01 0.1 | range of the member width and thickness (m) |

**Optional textures.** Put member textures (`*.png`, `*.jpg`) in `assets/textures/` and ground textures in `assets/backgrounds/`. When these folders are empty, random colours are used.

---

## Data format

Each domain (BridgeNet or synthetic) is a folder `<id>/<id>_render/` containing:

| File | Content |
|---|---|
| `img_<j>.jpg` | field image *j* (640 × 480) |
| `trans_<j>.npy`, `rots_<j>.npy` | camera position and world-to-camera rotation of image *j* (used for training only; not used at inference) |
| `line_bars.npy` | structural LD, `(N, 2, 3)`: the two end nodes of each bar |
| `pts.npy`, `sdf.npy` | query points and their signed distance to the member surfaces (training) |
| `Km_0.npy` (BridgeNet) / `Km.npy` (synthetic) | camera intrinsics |
| `dense_pts.npy` | dense ground-truth point cloud sampled from the CAD model (31 test domains) |

BridgeNet domains also include the CAD model (`<id>_models/<id>.obj`), the topology (`<id>_topology_properties/`) and the generator parameters (`<id>_input_params.json`).

---

## Data availability

| Item | Where |
|---|---|
| Code (network, training, evaluation, FE mesh export, synthetic data generator) | this repository |
| Pretrained BridgeNet model | release v1.0 (`mod_brdgnet_ptrain.pth`) |
| BridgeNet train/test data, test ids | release v1.0 (`part_000`–`part_011`), `data/bridgenet_test_ids.json` |
| Field data of the existing structures B-001–B-004 (images, LiDAR references, LDs) | not included in this release |

---

## Citation

If you use LiNeMesh or BridgeNet, please cite:

```bibtex
@article{linemesh,
  title   = {LiNeMesh: Line-Diagram-Informed Neural Mesh Generation from Multi-View Uncalibrated Images},
  author  = {To be added on publication},
  journal = {},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
