# run_voxel_cli.py
import argparse, subprocess, os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--pitch", type=float, default=0.2)
parser.add_argument("--tile_vox", type=int, default=256)
args = parser.parse_args()

# assume engine files are in ../src relative to this script (if running from local_client)
# or ../voxel_engine_src if synced by addon
# The addon syncs to "engine_src".
# If this script is inside "engine_src" (as implied by addon.py calling it there), it should work.
# But here I am creating it in "local_client".
# I will make it robust to find src.

script_dir = os.path.dirname(os.path.abspath(__file__))
# Check for src in parent
src_dir = os.path.join(script_dir, "..", "src")
if not os.path.exists(src_dir):
    # Check for engine_src (synced name)
    src_dir = os.path.join(script_dir, "..", "engine_src")

if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)
else:
    # If running from within src/engine_src
    sys.path.insert(0, script_dir)

try:
    from voxel_cuda import tile_voxelize
    import numpy as np
    import open3d as o3d
    from io_helpers import save_occ_npy
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# load input pointcloud
# We need loader from src
from loader import load_pointcloud, mesh_to_pointcloud

ext = args.input.split('.')[-1].lower()
if ext in ["obj", "stl", "fbx", "glb"]:
    pts, cols = mesh_to_pointcloud(args.input, n_samples=500000)
else:
    pts, cols = load_pointcloud(args.input)

print(f"Loaded {pts.shape[0]} points")

res = tile_voxelize(pts, pitch=args.pitch, tile_vox=args.tile_vox, use_gpu=True)

# Merge results
def merge_tiles(results, pitch):
    if not results:
        return np.zeros((1,1,1), dtype=bool), np.zeros(3)
    min_x = min([r['origin'][0] for r in results])
    min_y = min([r['origin'][1] for r in results])
    min_z = min([r['origin'][2] for r in results])
    max_x = max([r['origin'][0] + r['occ'].shape[0]*pitch for r in results])
    max_y = max([r['origin'][1] + r['occ'].shape[1]*pitch for r in results])
    max_z = max([r['origin'][2] + r['occ'].shape[2]*pitch for r in results])
    mn = np.array([min_x, min_y, min_z], dtype=np.float32)
    mx = np.array([max_x, max_y, max_z], dtype=np.float32)
    dims = np.ceil((mx-mn) / pitch).astype(int)
    grid = np.zeros(tuple(dims), dtype=bool)
    for r in results:
        occ = r['occ']
        origin = r['origin']
        off = np.floor((origin - mn) / pitch).astype(int)
        sx,sy,sz = occ.shape
        grid[off[0]:off[0]+sx, off[1]:off[1]+sy, off[2]:off[2]+sz] |= occ
    return grid, mn

merged, mn = merge_tiles(res, args.pitch)
save_occ_npy(args.output, merged)
print("Saved merged output to", args.output)
