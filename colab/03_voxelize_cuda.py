# %% [markdown]
# # Notebook C — 03_voxelize_cuda.ipynb (Fastest CUDA Voxelization)

# %%
# Cell C0 — Install dependencies
# !pip install numba numpy

# %%
# Cell C1 — Save voxel_cuda.py (GPU / CPU auto-switch)
voxel_cuda_code = r"""
# voxel_cuda.py
# 最速ボクセル化：CUDA（numba）またはCPUフォールバック

import numpy as np

# CUDA の有無チェック
try:
    from numba import cuda, njit, prange
    has_cuda = True
except Exception:
    from numba import njit, prange
    has_cuda = False

#----------------------------------------
# Bounding box
#----------------------------------------
def compute_bounds(points, padding=0.02):
    mn = points.min(axis=0).astype(np.float32)
    mx = points.max(axis=0).astype(np.float32)
    diag = np.linalg.norm(mx - mn)
    pad = padding * diag
    return (mn - pad).astype(np.float32), (mx + pad).astype(np.float32)

def create_dims(mn, mx, pitch):
    dims = np.ceil((mx - mn) / pitch).astype(np.int32)
    return int(dims[0]), int(dims[1]), int(dims[2])

#----------------------------------------
# CUDA 版（最速）
#----------------------------------------
if has_cuda:

    @cuda.jit
    def kernel_voxelize(points, mn, pitch, nx, ny, nz, counts):
        i = cuda.grid(1)
        if i >= points.shape[0]:
            return

        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        ix = int((x - mn[0]) / pitch)
        iy = int((y - mn[1]) / pitch)
        iz = int((z - mn[2]) / pitch)

        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            idx = ix * (ny * nz) + iy * nz + iz
            cuda.atomic.add(counts, idx, 1)

    def voxelize(points, pitch=0.3, thresh=1):
        mn, mx = compute_bounds(points)
        nx, ny, nz = create_dims(mn, mx, pitch)
        total = nx * ny * nz

        # GPU メモリ確保
        d_points = cuda.to_device(points.astype(np.float32))
        d_counts = cuda.to_device(np.zeros(total, dtype=np.uint32))

        # CUDA カーネル実行
        threads = 256
        blocks = (points.shape[0] + threads - 1) // threads
        kernel_voxelize[blocks, threads](d_points, mn, pitch, nx, ny, nz, d_counts)

        counts = d_counts.copy_to_host().reshape((nx, ny, nz))
        voxels = counts >= thresh
        return voxels, counts, mn, pitch

else:
    #----------------------------------------
    # CPUフォールバック（遅いが確実）
    #----------------------------------------
    @njit(parallel=True)
    def cpu_voxelize_inner(points, mn, pitch, nx, ny, nz):
        counts = np.zeros((nx, ny, nz), dtype=np.int32)
        for i in prange(points.shape[0]):
            x, y, z = points[i]

            ix = int((x - mn[0]) / pitch)
            iy = int((y - mn[1]) / pitch)
            iz = int((z - mn[2]) / pitch)

            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                counts[ix, iy, iz] += 1

        return counts

    def voxelize(points, pitch=0.3, thresh=1):
        mn, mx = compute_bounds(points)
        nx, ny, nz = create_dims(mn, mx, pitch)
        counts = cpu_voxelize_inner(points.astype(np.float32), mn, pitch, nx, ny, nz)
        voxels = counts >= thresh
        return voxels, counts, mn, pitch
"""
import os
SRC = "/content/voxel_engine_src"
os.makedirs(SRC, exist_ok=True)
with open(os.path.join(SRC, "voxel_cuda.py"), "w") as f:
    f.write(voxel_cuda_code)
print("Saved voxel_cuda.py")

# %%
# Cell C2 — Load module
import sys
if SRC not in sys.path:
    sys.path.append(SRC)

from voxel_cuda import voxelize, compute_bounds
import loader
import numpy as np
import os

# %%
# Cell C3 — Load input point cloud
BASE = "/content/drive/MyDrive/voxel_engine"
input_dir = os.path.join(BASE, "input")
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
files = os.listdir(input_dir)

print("Input candidates:", files)

if len(files) == 0:
    # raise RuntimeError("Please place point cloud or OBJ in input/")
    print("Warning: No input files found. Please upload files to run this cell.")
else:
    fp = os.path.join(input_dir, files[0])
    ext = fp.split('.')[-1].lower()

    if ext in ["obj", "stl", "fbx", "glb"]:
        pts, cols = loader.mesh_to_pointcloud(fp, n_samples=500000)
    else:
        pts, cols = loader.load_pointcloud(fp)

    print("Loaded:", pts.shape)

    # %%
    # Cell C4 — Execute CUDA / CPU Voxelization
    pitch = 0.30 # 30cm grid
    thresh = 1 # 1 point or more to fill

    vox, counts, mn, pitch = voxelize(pts, pitch=pitch, thresh=thresh)

    print("voxel grid:", vox.shape)
    print("occupied:", vox.sum())

    # %%
    # Cell C5 — Save voxels (.npy / JSON)
    import json
    out_dir = os.path.join(BASE, "output")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "voxels.npy"), vox.astype(np.uint8))

    meta = {
        "min_bounds": mn.tolist(),
        "pitch": float(pitch),
        "dims": list(vox.shape)
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved to output/")

    # %%
    # Cell C6 (Optional) — Visualization (Slice)
    import matplotlib.pyplot as plt
    import numpy as np

    slc = vox[:,:,vox.shape[2]//2]
    plt.imshow(slc)
    plt.title("mid slice (Z-plane)")
    plt.show()
