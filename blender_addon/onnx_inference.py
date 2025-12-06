import sys
import os

# Try to import from engine_src if available
addon_dir = os.path.dirname(__file__)
engine_src = os.path.abspath(os.path.join(addon_dir, "..", "engine_src"))
if os.path.exists(engine_src) and engine_src not in sys.path:
    sys.path.append(engine_src)

try:
    from classify import ONNXClassifier, compute_features
except ImportError:
    # Fallback or placeholder if engine_src is not yet synced
    class ONNXClassifier:
        def __init__(self, model_path):
            pass
        def predict(self, features):
            print("Engine not synced or classify module missing")
            return None
    def compute_features(points, colors=None, normals=None):
        return None

def run_inference(points, model_path):
    clf = ONNXClassifier(model_path)
    feats = compute_features(points)
    if feats is not None:
        return clf.predict(feats)
    return None
