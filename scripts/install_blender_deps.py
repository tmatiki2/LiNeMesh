"""
Install the extra Python packages that 01_LM_synthetic_data.py needs inside Blender's
own (bundled) Python interpreter. Run once, with Blender:

    blender -b --factory-startup -P scripts/install_blender_deps.py

Blender already ships numpy; this adds trimesh, scipy and rtree (used by trimesh to
test which sampled points lie inside the members).
"""
import subprocess
import sys

PKGS = ["trimesh", "scipy", "rtree"]
py = sys.executable
print("Blender Python:", py)
subprocess.call([py, "-m", "ensurepip", "--upgrade"])
subprocess.check_call([py, "-m", "pip", "install", "--upgrade", *PKGS])
for name in PKGS:
    __import__(name)
print("Installed and importable:", ", ".join(PKGS))
