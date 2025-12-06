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

    def voxelize_cuda(points, pitch=0.3, thresh=1, mn=None, mx=None):
        if mn is None or mx is None:
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
    
    # Alias for compatibility
    voxelize = voxelize_cuda

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

    def voxelize_cpu(points, pitch=0.3, thresh=1, mn=None, mx=None):
        if mn is None or mx is None:
            mn, mx = compute_bounds(points)
        nx, ny, nz = create_dims(mn, mx, pitch)
        counts = cpu_voxelize_inner(points.astype(np.float32), mn, pitch, nx, ny, nz)
        voxels = counts >= thresh
        return voxels, counts, mn, pitch

    # Alias for compatibility
    voxelize = voxelize_cpu

# Tile wrapper (1D tiling on X axis; can be extended)
def tile_voxelize(points, pitch=0.3, tile_vox=256, thresh=1, use_gpu=True):
    mn, mx = compute_bounds(points)
    tile_size = tile_vox * pitch
    x_tiles = int(np.ceil((mx[0]-mn[0]) / tile_size))
    results = []
    for tx in range(x_tiles):
        x0 = mn[0] + tx * tile_size
        x1 = x0 + tile_size
        mask = (points[:,0] >= x0) & (points[:,0] < x1)
        pts_tile = points[mask]
        if pts_tile.shape[0] == 0:
            continue
        tmn = np.array([x0, mn[1], mn[2]], dtype=np.float32)
        tmx = np.array([x1, mx[1], mx[2]], dtype=np.float32)
        
        if use_gpu and has_cuda:
            occ, counts, origin, _ = voxelize_cuda(pts_tile, pitch=pitch, thresh=thresh, mn=tmn, mx=tmx)
        else:
            # If CUDA is not available, fall back to CPU even if use_gpu is True
            occ, counts, origin, _ = voxelize(pts_tile, pitch=pitch, thresh=thresh, mn=tmn, mx=tmx)
            
        results.append({'tile_index':tx, 'occ':occ, 'counts':counts, 'origin':origin})
    return results
