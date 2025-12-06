# Voxelizer Engine

ColabでCUDA最適化のボクセル化エンジンを開発 → GitHub配布 → Blender Add-onが参照してローカル実行するワークフロー。

## Quickstart

1. Open `colab/01_load_points.py` (or .ipynb) in Google Colab, run and develop.
2. Push changes to GitHub (use PAT).
3. In Blender, install `blender_addon/addon.py` and use the "Sync" button to pull the latest engine.

## Directory Structure

- `colab/`: Colab notebooks for development and training.
- `src/`: Core python modules (loader, voxelizer, classifier).
- `blender_addon/`: Blender addon to sync and run the engine.
- `local_client/`: CLI wrapper for local execution.
