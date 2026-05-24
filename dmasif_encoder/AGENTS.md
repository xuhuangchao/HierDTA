# dMaSIF ENCODER KNOWLEDGE BASE

**Generated:** 2026-05-24
**Parent:** E:\AIDD分子设计\HierDTA

## OVERVIEW

Vendored dMaSIF surface geometry encoder for protein binding pockets. Extracts oriented point cloud embeddings via PyKeOps-accelerated geometric convolutions and atom-type chemical features.

## STRUCTURE

```
dmasif_encoder/
├── protein_surface_encoder.py   # dMaSIF class + AtomNet variants — surface embedding extraction (277 lines)
├── geometry_processing.py       # Point cloud ops — subsample, curvatures, normals, atom coords, dMaSIFConv kernel (730 lines)
├── data_iteration.py            # Batched surface processing — iterate() and iterate_surface_precompute() (75 lines)
├── helper.py                    # Tensor type selectors + diagonal_ranges/ranges_slices utilities (34 lines)
└── benchmark_models.py          # dMaSIFConv_seg reference model with oriented tangent frame steering (175 lines)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Surface embedding forward pass | `protein_surface_encoder.py:258` | dMaSIF.forward() — features() + conv.load_mesh() + conv() |
| Geometric feature extraction | `geometry_processing.py` | curvatures(), subsample(), atoms_to_points_normals() |
| Mesh convolution kernel | `geometry_processing.py` | dMaSIFConv class — KeOps LazyTensor gaussian kernel |
| Tangent frame orientation | `benchmark_models.py:87` | load_mesh() — steered uv basis from weight gradients |
| Batch-aware ranges | `helper.py:22` | diagonal_ranges() — KeOps block-diagonal structure for batches |
| Pocket selection logic | `data_iteration.py:62` | select_pocket() — 512 nearest points to ligand center |
| dMaSIF config parameters | `../config.yml` | curvature_scales, resolution, radius, emb_dims, atom_dims, n_layers, dropout |

## CONVENTIONS

- **Docstrings**: Mixed. `geometry_processing.py` uses Google-style (Args:/Returns:). `protein_surface_encoder.py` uses brief one-liners. `benchmark_models.py` uses custom bullet format for `load_mesh()`.
- **No type hints anywhere** — zero annotations across all 5 files.
- **No `logging` module** — all output via `print()` and `tqdm`.
- **PyKeOps** is the backbone: `LazyTensor` for kernel ops, `grid_cluster` for subsampling, `diagonal_ranges` for batch isolation.
- **No `__init__.py`** — directory is not a package. Imported via sys.path manipulation from root scripts (`surface_process.py`, `create_data_motif.py`).
- **GPU-suffixed tensor type selectors** in `helper.py` lines 4-5 — NOT device-aware, checked at import time only.
- **Naming**: PascalCase classes (`dMaSIF`, `AtomNet_MP`), snake_case functions (`knn_atoms`, `diagonal_ranges`).
- **`config.yml` at root** controls all dMaSIF hyperparams. No argparse in this directory.

## ANTI-PATTERNS

| Issue | Location | Severity |
|-------|----------|----------|
| `from logging import raiseExceptions` | `geometry_processing.py:1` | Bug — `raiseExceptions` does not exist in `logging` module. Latent ImportError if triggered. |
| `tensor = torch.cuda.FloatTensor if torch.cuda.is_available()` | `helper.py:4-5` | Anti-pattern — evaluated at import time, not at runtime. Fails if GPU state changes or device is specified later. |
| Commented-out imports (matplotlib, pyvtk) | `geometry_processing.py:9,17` | Dead code |
| Commented-out model layers (3 blocks) | `benchmark_models.py:19-47` | Code rot — old 3-layer architecture left commented |
| Wildcard import `from dmasif_encoder.helper import *` | `geometry_processing.py:5` | Unclear dependencies |
| `to_dense_batch` import from PyG | `data_iteration.py:6` | Only PyG dependency in this directory — everything else is raw PyTorch + KeOps |

## UNIQUE STYLES

- **KeOps LazyTensor** for differentiable geometric kernels — pairwise distance, gaussian window, orientation steering. NOT standard PyTorch.
- **Block-diagonal batch processing** — `diagonal_ranges()` assigns `.ranges` to LazyTensor to prevent cross-batch interactions, no padding/packing.
- **grid_cluster subsampling** — cubic grid cells (scale param in Angstroms) for uniform point cloud downsampling, NOT FPS or random.
- **Multi-scale curvature features** — scalar mean/Gaussian curvatures computed at `curvature_scales` (list of floats) via tangent plane PCA.
- **Atom-type chemical features** — 22 atom types (residue-level grouping) projected through AtomNet message passing, then concatenated with curvatures.
- **Oriented tangent frame steering** — `load_mesh()` re-orients uv bases using the spatial gradient of predicted orientation weights, making convolution rotation-invariant.
- **Dict-based data passing** — `dMaSIF.forward()` takes and returns a dict `P` with keys: `xyz`, `normals`, `batch`, `atom_xyz`, `atomtypes`, `batch_atoms`, `input_features`, `embedding`. No dataclasses.
- **Pocket truncation to 512 points** — `select_pocket()` at `data_iteration.py:73` hardcodes `point_nums = 512` nearest points to ligand center. Affects all downstream surface features.
