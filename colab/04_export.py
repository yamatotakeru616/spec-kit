# %% [markdown]
# # Notebook D — 04_export.ipynb (Post-processing & Export)

# %%
# Cell D0 — Install dependencies
# !pip install numpy scipy scikit-image open3d trimesh py-vox-io matplotlib

# %%
# Cell D1 — Setup & Imports
import os, sys, json, math
BASE = "/content/drive/MyDrive/voxel_engine"
SRC = "/content/voxel_engine_src"
if os.path.exists(SRC) and SRC not in sys.path:
    sys.path.append(SRC)

import numpy as np
import open3d as o3d
import trimesh
from skimage import measure
from scipy import ndimage
from pyvox.models import Vox
from pyvox.writer import VoxWriter

print("BASE:", BASE)
os.makedirs(os.path.join(BASE,"output"), exist_ok=True)

# %%
# Cell D2 — Load Voxels
# Try to load merged voxels if exist, else look for individual tile results in runtime variable `res`
out_npy = os.path.join(BASE, "output", "voxels.npy") # Changed from merged_voxel.npy to voxels.npy to match previous notebook
if os.path.exists(out_npy):
    vox = np.load(out_npy).astype(bool)
    mn = None
    # if meta exists
    meta_path = os.path.join(BASE,"output","meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path,'r'))
        mn = np.array(meta.get("min_bounds", [0,0,0]), dtype=np.float32)
        pitch = float(meta.get("pitch", 0.2))
    else:
        pitch = 0.2
    print("Loaded voxels.npy:", vox.shape)
else:
    print("voxels.npy not found. Please run Notebook C first.")
    # Create dummy data for testing if file missing
    vox = np.zeros((10,10,10), dtype=bool)
    mn = np.array([0,0,0], dtype=np.float32)
    pitch = 0.2

# %%
# Cell D3 — Morphology (Noise Removal & Hole Filling)
# binary closing/opening and small hole fill
print("Original occupied voxels:", vox.sum())
# Convert to uint8 for ndimage
arr = vox.astype(np.uint8)

# 3D binary closing to fill small gaps
closed = ndimage.binary_closing(arr, iterations=1)

# remove small isolated components
labeled, num = ndimage.label(closed)
sizes = ndimage.sum(closed, labeled, range(1, num+1))
min_size = 10 # voxels
mask = np.zeros_like(closed)
for i, s in enumerate(sizes, start=1):
    if s >= min_size:
        mask[labeled == i] = True

clean = mask.astype(bool)
print("After morph -> occupied:", clean.sum())

# Save cleaned result
np.save(os.path.join(BASE,"output","voxels_clean.npy"), clean.astype(np.uint8))
vox = clean # use cleaned for further steps

# %%
# Cell D4 — Semantic/Color Bake (Optional)
# If you have per-point semantic labels and colors, aggregate them into voxels.
# Assumes `pts`, `labels`, `colors` are in runtime (e.g., from classify step).
if 'pts' in globals() and 'labels' in globals():
    print("Aggregating semantic labels to voxels ...")
    # Implementation skipped as variables likely not in scope across notebooks without saving/loading
    pass
else:
    print("No per-point labels found in runtime; skipping semantic bake.")

# %%
# Cell D5 — Marching Cubes (Mesh Extraction) & Save (OBJ / PLY / GLB)
# marching cubes requires volumetric data as float (1 inside, 0 outside)
vol = vox.astype(np.uint8)
# skimage.measure.marching_cubes expects axes (z,y,x) often; we will pass vol.transpose to get consistent orientation if needed.
try:
    if vol.sum() > 0:
        verts, faces, normals, values = measure.marching_cubes(vol.transpose(2,1,0), level=0.5) # transpose to (Z,Y,X)
        # convert voxel indices back to world coords
        # Note: verts are in voxel index space; mapping to world: world = mn + (idx + 0.5) * pitch
        verts_world = []
        for v in verts:
            z_idx, y_idx, x_idx = v # since we transposed
            xw = mn[0] + (x_idx + 0.5) * pitch
            yw = mn[1] + (y_idx + 0.5) * pitch
            zw = mn[2] + (z_idx + 0.5) * pitch
            verts_world.append((xw, yw, zw))
        verts_world = np.array(verts_world, dtype=np.float32)
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals, process=False)
        # save
        out_obj = os.path.join(BASE,"output","mesh_from_vox.obj")
        mesh.export(out_obj)
        print("Saved mesh OBJ to", out_obj)
        # also save GLB (binary glTF)
        out_glb = os.path.join(BASE,"output","mesh_from_vox.glb")
        mesh.export(out_glb)
        print("Saved GLB to", out_glb)
    else:
        print("Volume empty, skipping marching cubes.")
except Exception as e:
    print("Marching cubes failed:", e)

# %%
# Cell D6 — MagicaVoxel (.vox) Export
# Write .vox using py-vox-io
out_vox = os.path.join(BASE,"output","merged.vox")
try:
    # py-vox-io expects dense array shape (x,y,z). Ensure type uint8.
    dense = vox.astype(np.uint8)
    # If dims too large for MagicaVoxel (usually <256), consider downsample or tiling.
    if dense.shape[0] > 256 or dense.shape[1] > 256 or dense.shape[2] > 256:
        print("Large dims:", dense.shape, "Consider downsampling before Vox export.")
    
    if dense.sum() > 0:
        vox_model = Vox.from_dense(dense)
        VoxWriter(out_vox).write(vox_model)
        print("Saved .vox to", out_vox)
    else:
        print("Volume empty, skipping vox export.")
except Exception as e:
    print("Saving .vox failed:", e)

# %%
# Cell D7 — Open3D Preview
# show voxel points (downsample if too many)
idx = np.argwhere(vox)
if idx.shape[0] > 0:
    pts = idx * pitch + mn + pitch/2.0
    print("voxel points:", pts.shape)
    max_pts = 200000
    if pts.shape[0] > max_pts:
        sel = np.random.choice(pts.shape[0], max_pts, replace=False)
        pts_vis = pts[sel]
    else:
        pts_vis = pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_vis)
    # o3d.visualization.draw_geometries([pcd]) # Cannot run GUI in headless colab usually
    print("Created Open3D pointcloud for visualization")
else:
    print("No voxels to visualize")

# %%
# Cell D8 — Completion Log
print("Export finished. Files in:", os.path.join(BASE,"output"))
if os.path.exists(os.path.join(BASE,"output")):
    for fn in sorted(os.listdir(os.path.join(BASE,"output"))):
        print("-", fn)
