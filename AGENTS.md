# AGENTS.md

## What is this

Research code for the ECCV 2024 paper "REFRAME: Reflective Surface Real-Time Rendering for Mobile Devices". Real-time novel-view synthesis of reflective surfaces using mesh-based rendering with neural environment maps.

## Environment

- Python 3.12, PyTorch 2.2.1, CUDA 12.1
- Requires `tiny-cuda-nn` (imported as `tinycudann`) and `nvdiffrast` — both must be built from source before `pip install -r requirements.txt`
- Conda env name: `REFRAME` (use mamba for faster installs)

## Running

Single entrypoint: `main.py` via CLI args.

```bash
# Train
python main.py --datadir <data_path> --initial_mesh <mesh.obj> --run_name <name>

# Test + UV map
python main.py --datadir <data_path> --initial_mesh <trained_mesh> --run_name <name> \
  --shader_path <shader.pt> --test 1 --uvmap 1
```

Output lands in `./output/<run_name>/` (images, meshes, shaders, tensorboard logs, code backup).

## Key flags

- `--dataset blender|colmap` — blender for object scenes, colmap for open scenes
- `--wenvlearner 0` — disables env learner, optimizes envmap directly (faster, lower quality)
- `--refneus 1` — use Ref-NeuS initial mesh (requires `points_of_interest.ply` in data dir)
- `--scale 0.7` — required for hotdog/ship scenes (default 0.8)
- `--region 1|2` — 1=object, 2=open scene
- `--ssaa 2` — super sampling rate
- `--uvmap 1` — run UV mapping after test

## Package layout (`reframe/`)

- `mesh.py` — Mesh class (vertices, indices, normals)
- `surfacerenderer.py` — nvdiffrast-based rasterizer with deferred shading
- `tcnnshader.py` — hash-grid encoder + MLP shader (diffuse+specular decomposition)
- `velearner.py` — vertex offset network (hash grid + MLP)
- `norlearner.py` — normal offset network (hash grid + frequency encoding)
- `blenderdataset.py` — NeRF Synthetic / Shiny Blender loader
- `colmapdataset.py` — COLMAP format loader (binary or text)
- `utils.py` — mesh I/O, UV mapping, envmap baking, MLP class

## Gotchas

- No tests, no lint, no typecheck — this is a research repo
- `nvdiffrast` uses `RasterizeGLContext` by default; if OpenGL fails, switch to `RasterizeCudaContext` in `surfacerenderer.py` and `utils.py`
- Seed is hardcoded to 0 (`utils.seed_everything(0)`)
- `torch.autograd.set_detect_anomaly(True)` is on — slows training but catches NaN
- Training auto-triggers test+uvmap after final epoch
- `setuptools<71` required for tiny-cuda-nn build (newer versions removed `pkg_resources`)
- `numpy<2` required for PyTorch 2.2.x compatibility
