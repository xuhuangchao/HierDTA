# HierDTA — Project Knowledge Base

**Generated:** 2025-05-26
**Commit:** b377a68
**Branch:** main

## OVERVIEW

HierDTA: Drug-Target Affinity (DTA) prediction via hierarchical GNNs.
Drug: AttentiveFP (PyG) on molecular graphs. Protein: EGNN on 3D residue-level pocket graphs with ESM-2 + dMaSIF surface features. Interaction: bilinear co-attention + MLP decoder.
Stack: Python 3.10, PyTorch 2.5 + PyG 2.6, RDKit, fair-esm, MDAnalysis.

## STRUCTURE

```
HierDTA/
├── models/                    # All model architecture (5 files)
│   ├── model.py               #   HierDTA top-level + CoAttentionModule + MLPDecoder
│   ├── drug_model.py          #   DrugModel wrapper → AttentiveFP
│   ├── attentivefp.py         #   GATEConv + GATConv with GRU gating
│   ├── protein_model.py       #   ProteinEGNN wrapper → EGNN
│   └── egnn_clean.py          #   E_GCL layer + EGNN stack
├── dmasif_encoder/            # dMaSIF surface geometry (5 files)
│   ├── protein_surface_encoder.py  #   dMaSIF model (AtomNet_MP + conv)
│   └── geometry_processing.py      #   Curvature computation
├── initial/                   # Embedding dumps (pickle + txt)
│   ├── ele2emb.pkl            #   ElementKG 133-dim atom embeddings
│   ├── fg2emb.pkl / rel2emb.pkl  #   (legacy/unused)
│   └── get_dict.py            #   Dumps pickles from text
├── data/                      # Raw datasets + preprocessed + caches
├── config.yml                 # dMaSIF model config only
├── training_warmup.py         # Main training entry point
├── run_preprocessing.py       # One-click preprocessing pipeline
├── split_all_dataset.py       # Step 1: dataset splitting
├── protein_process.py         # Step 2: ESM-2 extraction
├── surface_process.py         # Step 3: dMaSIF surface → residue mapping
├── create_data_motif.py       # Step 4: global cache assembly
├── utils.py                   # TestbedDatasetHMol + GPU metrics
├── chemutils.py               # RDKit mol graph construction
├── summarize_results.py       # Mean/std aggregation across seeds
├── train.sh / train2.sh       # Seed-loop training scripts
├── experiments.txt            # Experiment plan & ablation checklist
└── environment.yml            # Conda env spec (name: pymesh)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify drug GNN layer | `models/attentivefp.py` | GATEConv + GATConv stack; node dims flow through `DrugModel` |
| Add/modify protein encoder | `models/egnn_clean.py`, `models/protein_model.py` | EGNN layers in egnn_clean; feature slicing (41-dim strip) in protein_model |
| Change interaction mechanism | `models/model.py` → `CoAttentionModule` | Cross-attn + bilinear fusion; `forward()` returns `(logits, output_dict)` |
| Add new evaluation metric | `utils.py` → GPU metric functions | rmse_gpu, mse_gpu, pearson_gpu, ci_gpu, get_rm2_gpu |
| Add new CLI argument | `training_warmup.py` → argparse block (lines 13-54) | Args flow: CLI → `HierDTA.__init__()` → sub-modules |
| Change dataset splitting | `split_all_dataset.py` | 4 strategies: random, cold_drug, cold_target, all_cold; 5 seeds |
| Add new preprocessing step | `run_preprocessing.py` | Smart-skip logic with `check_*_exists()` helpers |
| Modify molecular featurization | `chemutils.py` | one-of-k encoding, BRICS motif decomposition |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `HierDTA` | class | `models/model.py:160` | Top-level model: drug_net + protein_net + interaction |
| `CoAttentionModule` | class | `models/model.py:36` | Bidirectional cross-attn + bilinear fusion |
| `MLPDecoder` | class | `models/model.py:24` | 128→256→1 MLP predictor |
| `AttentiveFP` | class | `models/attentivefp.py:67` | Drug GNN (GATEConv + GATConv) |
| `GATEConv` | class | `models/attentivefp.py:13` | Single attention-based message passing layer |
| `DrugModel` | class | `models/drug_model.py:9` | Drug encoder wrapper |
| `EGNN` | class | `models/egnn_clean.py:106` | E(n)-Equivariant GNN stack |
| `E_GCL` | class | `models/egnn_clean.py:5` | Single equivariant conv layer |
| `ProteinEGNN` | class | `models/protein_model.py:6` | Protein encoder wrapper; handles feature slicing |
| `dMaSIF` | class | `dmasif_encoder/protein_surface_encoder.py:207` | Surface geometry encoder |
| `TestbedDatasetHMol` | class | `utils.py:10` | PyG Dataset; loads global caches, assembles data pairs |
| `train()` | func | `training_warmup.py:57` | Single-epoch training loop |
| `predicting_gpu()` | func | `training_warmup.py:78` | Eval loop; returns `(labels, preds, attention_dict)` |

## CONVENTIONS

- **No `__init__.py`**: Models imported via relative imports (`from .drug_model import *`). Direct file imports work from root.
- **Model outputs**: `training=True` → returns affinity only. `training=False` → returns `(affinity, output_dict)` with attention records.
- **Feature dimensions are hard constraints**: `drug_out` and `protein_out` MUST equal `emb_dim` (asserted in `HierDTA.__init__`).
- **Protein feature layout**: Raw node features = [41 physchem | 480 ESM-2 | 128 surface]. `ProteinEGNN` strips physchem (keeps `h[:, 41:]` with surface, or `h[:, 41:521]` without).
- **Global caches** in `data/cache/` are seed/strategy-independent; loaded once in `TestbedDatasetHMol.__init__`.
- **Results saved per run** to `results_{dataset}/{strategy}/seed_{seed}/` — checkpoints + test metrics + attention records.
- **GPU-first**: All metrics (`utils.py` `*_gpu` functions) operate on GPU tensors. CPU fallback functions also exist (unused in training).
- **No type hints**: The codebase uses no Python type annotations (except AttentiveFP's docstring args).
- **`training_warmup.py` has no `if __name__ == "__main__"` guard** — top-level code runs unconditionally at import. Do not import this file as a module.
- **Chinese comments**: Some inline comments and argparse descriptions are in Chinese — keep ASCII for new code.
- **Conda env**: `pymesh` with Tsinghua mirrors; PyTorch built for CUDA 12.4.

## ANTI-PATTERNS (THIS PROJECT)

- **NEVER change feature dimension layout** without updating: `chemutils.py` atom featurizer, `protein_process.py` ESM dims, `ProteinEGNN.forward()` slicing logic, and `HierDTA.__init__` args.
- **NEVER add dependency** without updating `environment.yml` — the env uses pinned versions and Chinese mirrors.
- **NEVER use `from models import *` across directories** — relative imports only within `models/`.
- **NEVER modify `attentivefp.py` GATEConv signature** — it's a PyG MessagePassing subclass with strict method signatures.
- **NEVER import `training_warmup.py` as a module** — it has no `if __name__ == "__main__"` guard; importing triggers training.

## UNIQUE STYLES

- **`EDGE_DIM = 14`** is hardcoded in `drug_model.py` (not configurable). Bond feature dimensionality is Chemprop-standard.
- **`num_timesteps=2`** hardcoded in `DrugModel.__init__`; default AttentiveFP readout iterations.
- **`weights_only=False`** on all `torch.load()` calls — required for PyG data objects with custom classes.
- **EGNN uses `unsorted_segment_sum/mean`** — custom CUDA scatter ops, not PyG's `scatter`.
- **CoAttention reuses `scores_d2p.transpose`** for p2d attention instead of recomputing.
- **`compute_metrics_gpu()`** returns list `[rmse, mse, pearson, ci, rm2]` — positional, not dict. Best model selected by `val_ret[1]` (MSE).
- **dMaSIF config (`config.yml`)** is separate from training config (which uses only argparse).

## COMMANDS

```bash
# Environment
conda env create -f environment.yml
conda activate pymesh

# Full preprocessing pipeline (steps 1-4)
python run_preprocessing.py --dataset kiba --k_list 3,5,8 --gpu_idx 1

# Single training run
python training_warmup.py --dataset_idx 1 --surface_k 5 --gpu_idx 1 --seed 1 --strategy random

# Multi-seed batch training (cold_drug + cold_target)
bash train.sh    # cold_drug, gpu 1
bash train2.sh   # cold_target, gpu 0

# Results aggregation
python summarize_results.py --base_dir results_kiba --strategy random
```

## NOTES

- **Python 3.10 only** — `match`/`case`, `|` union types, and other 3.10+ features may not work.
- **dMaSIF requires KeOps** (`pykeops`) — custom CUDA kernels, may fail on some GPU architectures.
- **Initial embeddings** (`initial/`) are loaded by `surface_process.py` and `chemutils.py` via pickle; paths may need adjustment on different machines.
- **Preprocessing assumes specific file structure** (`data/{dataset}/pocket1_{dataset}/*.pdb`, `process.csv`, `proteins.txt`). Missing files cause silent errors.
- **No test suite** — validate changes by running a single-epoch training on a subset.
- **`experiments.txt`** contains the authoritative experiment plan — check before modifying architecture.
