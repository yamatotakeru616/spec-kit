# loader.py - point & mesh loader + preprocessing
import numpy as np
import open3d as o3d
import trimesh
import laspy

def load_pointcloud(path):
    ext = path.split('.')[-1].lower()
    if ext in ['ply','pcd','xyz']:
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if np.asarray(pcd.colors).size else None
        return points, colors
    elif ext in ['las','laz']:
        las = laspy.read(path)
        pts = np.vstack([las.x, las.y, las.z]).T
        return pts.astype(np.float32), None
    else:
        raise ValueError('Unsupported pointcloud format: ' + ext)

def mesh_to_pointcloud(path, n_samples=200000):
    mesh = trimesh.load(path, force='mesh')
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float32), None

def normalize_bounds(points, padding=0.02):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    pad = padding * diag
    return (mn - pad).astype(np.float32), (mx + pad).astype(np.float32)
