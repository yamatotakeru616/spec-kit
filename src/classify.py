# classify.py - feature extractor + ONNX wrapper
import numpy as np
import onnxruntime as ort

def compute_normals(points, k=30):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals, dtype=np.float32)

def compute_features(points, colors=None, normals=None):
    # features: [x,y,z, nx,ny,nz, r,g,b, height]
    centroid = points.mean(axis=0)
    scale = np.max(np.linalg.norm(points - centroid, axis=1))
    xyz_norm = (points - centroid) / (scale + 1e-9)
    if normals is None:
        normals = compute_normals(points)
    feats = [xyz_norm, normals]
    if colors is not None:
        feats.append(colors)
    # height normalized
    height = (points[:,2:3] - points[:,2].min()) / (points[:,2].ptp() + 1e-9)
    feats.append(height)
    feature = np.concatenate(feats, axis=1).astype(np.float32)
    return feature

class ONNXClassifier:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, features, batch_size=8192):
        N = features.shape[0]
        outputs = []
        for i in range(0, N, batch_size):
            batch = features[i:i+batch_size]
            # try both shapes
            try:
                out = self.sess.run(None, {self.input_name: batch})
            except Exception:
                out = self.sess.run(None, {self.input_name: batch[np.newaxis,...]})
            outputs.append(out[0])
        return np.vstack(outputs)
