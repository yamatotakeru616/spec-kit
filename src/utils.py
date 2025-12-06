# utils.py - small helpers
import numpy as np
import os
def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def save_json(path, obj):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
