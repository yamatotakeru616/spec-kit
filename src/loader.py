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

def load_annotated_pointcloud(path):
    """
    Loads point cloud and tries to extract labels (classification).
    Returns: points, colors, labels
    """
    ext = path.split('.')[-1].lower()
    
    if ext in ['las', 'laz']:
        las = laspy.read(path)
        pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        
        # Colors
        colors = None
        if hasattr(las, 'red'):
            # Normalize 16-bit colors to 0-1
            red = np.array(las.red) / 65535.0
            green = np.array(las.green) / 65535.0
            blue = np.array(las.blue) / 65535.0
            colors = np.vstack([red, green, blue]).T.astype(np.float32)
            
        # Labels
        labels = None
        if hasattr(las, 'classification'):
            labels = np.array(las.classification).astype(np.uint8)
            
        return pts, colors, labels
        
    elif ext in ['ply', 'pcd', 'xyz']:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if np.asarray(pcd.colors).size else None
        # Open3D doesn't easily load custom scalar fields like labels from PLY without tensor API
        # verify if we can get it, otherwise return None for labels
        return points, colors, None
    else:
        # Fallback for meshes
        pts, colors = mesh_to_pointcloud(path)
        return pts, colors, None
