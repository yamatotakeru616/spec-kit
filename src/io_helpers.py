import numpy as np
import os
import open3d as o3d
from pyvox.models import Vox
from pyvox.writer import VoxWriter

def save_occ_npy(path, occ):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, occ.astype(np.uint8))

def occ_to_pointcloud(occ, origin, pitch):
    idx = np.argwhere(occ)
    pts = idx * pitch + origin + pitch/2.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def save_vox(path, occ):
    # py-vox-io expects dense 3d array; ensure type conversion
    vox = Vox.from_dense(occ.astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    VoxWriter(path).write(vox)
