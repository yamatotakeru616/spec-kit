# %% [markdown]
# # Notebook B — 02_classify.ipynb (Feature Extraction + ONNX Inference)

# %%
# Cell B0 — Save classify.py module
# %%bash
# cat > /content/voxel_engine_src/classify.py <<'PY'
classify_code = r"""
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
"""
import os
SRC = "/content/voxel_engine_src"
os.makedirs(SRC, exist_ok=True)
with open(os.path.join(SRC, "classify.py"), "w") as f:
    f.write(classify_code)
print("wrote classify.py")

# %%
# Cell B1 — Sample execution (Load -> Features -> ONNX Inference Stub)
import sys
if SRC not in sys.path:
    sys.path.append(SRC)
import loader, classify, utils
import numpy as np, os

# pick input file (place one in BASE/input)
BASE = "/content/drive/MyDrive/voxel_engine"
input_dir = os.path.join(BASE,"input")
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
files = os.listdir(input_dir)
print("input files:", files)

# choose first file
if len(files)==0:
    print("No files in input/ - upload a sample PLY or OBJ")
else:
    fp = os.path.join(input_dir, files[0])
    ext = fp.split('.')[-1].lower()
    if ext in ['obj','glb','fbx','stl']:
        pts,cols = loader.mesh_to_pointcloud(fp, n_samples=200000)
    else:
        pts,cols = loader.load_pointcloud(fp)
    print("loaded", pts.shape)
    
    feats = classify.compute_features(pts, colors=cols, normals=None)
    print("features shape:", feats.shape)
    
    # ONNX model path placeholder:
    model_path = os.path.join(BASE, "models", "model.onnx")
    if os.path.exists(model_path):
        clf = classify.ONNXClassifier(model_path)
        out = clf.predict(feats, batch_size=8192)
        print("pred shape", out.shape)
    else:
        print("No model.onnx found at", model_path, "- skip infer")
