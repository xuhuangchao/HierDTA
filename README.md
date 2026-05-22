# HierDTA: Hierarchical Drug-Target Affinity Prediction

HierDTA is a deep learning framework for **Drug-Target Affinity (DTA)** prediction that leverages hierarchical molecular representations. It models drugs as heterogeneous graphs containing both **atoms** and **motifs** (BRICS fragments with functional group semantics), and proteins as 3D residue graphs augmented with surface geometry and language model embeddings.

---

## Architecture Overview

HierDTA consists of three main components:

### 1. Drug Encoder — `DrugMotifGAT`

A unified **GATv2** graph neural network processes atoms and motifs in a single heterogeneous graph.

- **Atom nodes**: Chemprop standard 133-dimensional features (atomic number, degree, formal charge, chirality, hybridization, aromaticity, mass).
- **Motif nodes**: BRICS decomposition fragments enriched with 133-dimensional **fg2emb** functional group embeddings. Motifs are matched against 82 functional groups via SMARTS patterns.
- **Edges**:
  - Atom–Atom edges: chemical bonds (bond type, ring membership).
  - Atom–Motif edges: **bidirectional** membership links, enabling both bottom-up (atom → motif) and top-down (motif → atom) message passing.
- **Architecture**: 2-layer GATv2 with multi-head attention (heads=2), hidden dim=64, output dim=128.

### 2. Protein Encoder — `ProteinEGNN`

An **E(n)-Equivariant Graph Neural Network** encodes the protein binding pocket.

- **Residue node features (608-dim)**:
  - **480-dim ESM-2**: per-residue embeddings from `esm2_t12_35M_UR50D` (layer 12).
  - **128-dim surface geometry**: extracted by the **dMaSIF** encoder from protein surface point clouds and mapped to residues via KDTree (k nearest neighbors, configurable).
- **Edges**: constructed with a 10.0 Å distance cutoff between residue CA atoms (or geometric centers).

### 3. Dual Interaction Module

A cross-attention mechanism models drug–protein interactions at multiple levels.

- **Atom→Protein (a2p)** cross-attention.
- **Motif→Protein (m2p)** cross-attention.
- **Fingerprint** (ECFP, 1024-dim) projected and fused.
- **Global protein representation** (mean-pooled ESM-2, 480-dim) projected and fused.
- Final affinity prediction via an MLP decoder.

---

## Directory Structure

```
HierDTA/
├── data/
│   ├── davis/                  # Davis dataset raw files
│   │   ├── process.csv
│   │   ├── proteins.txt
│   │   └── pocket1_davis/      # Pocket PDB files
│   ├── kiba/                   # KIBA dataset raw files
│   │   ├── process.csv
│   │   ├── proteins.txt
│   │   └── pocket1_kiba/
│   ├── cache/                  # Global feature caches
│   │   ├── {dataset}_smile_graph.pt
│   │   ├── {dataset}_fingerprint.pt
│   │   ├── {dataset}_protein_graphs_k{k}.pt
│   │   └── {dataset}_esm_feats.pt
│   └── processed/              # (legacy, no longer used for storage)
├── models/
│   ├── drug_model.py           # DrugMotifGAT
│   ├── protein_model.py        # ProteinEGNN
│   └── model.py                # HierDTA + DualInteractionModule
├── dmasif_encoder/             # dMaSIF surface encoding module
├── initial/
│   └── fg2emb.pkl              # Functional group knowledge embeddings
├── split_data/                 # Train/val/test CSV splits
├── split_all_dataset.py        # Step 1: dataset splitting
├── protein_process.py          # Step 2: ESM-2 sequence features
├── surface_process.py          # Step 3: surface feature extraction
├── create_data_motif.py        # Step 4: build global caches
├── training_warmup.py          # Step 5: training
├── run_preprocessing.py        # One-click preprocessing pipeline
└── utils.py                    # TestbedDatasetHMol (in-memory, no .pt saving)
```

---

## Installation

### Environment

The project is developed with Python 3.10 and PyTorch 2.5. We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate pymesh
```

### Key Dependencies

- `torch` >= 2.5, `torch-geometric` >= 2.6
- `rdkit` (molecular graph construction)
- `MDAnalysis` (protein structure processing)
- `fair-esm` (ESM-2 protein language model)
- `dmasif_encoder` (protein surface geometry, included in repo)
- `Biopython`, `scipy`, `networkx`, `pandas`, `numpy`

---

## Data Preparation

Download and place the DTA datasets under `data/`:

- **Davis**: `data/davis/process.csv`, `data/davis/proteins.txt`, `data/davis/pocket1_davis/*.pdb`
- **KIBA**: `data/kiba/process.csv`, `data/kiba/proteins.txt`, `data/kiba/pocket1_kiba/*.pdb`

`process.csv` must contain at least the columns: `Drug` (SMILES), `target_key`, `Y` (affinity).

---

## Usage Order

### Option A: One-Click Preprocessing

Run the full preprocessing pipeline for a chosen dataset and surface mapping parameter `k`:

```bash
python run_preprocessing.py \
  --dataset kiba \
  --k_list 3,5,8 \
  --gpu_idx 1
```

This executes **Steps 1–4** automatically with smart skip logic (already-computed outputs are not reprocessed).

### Option B: Step-by-Step Preprocessing

#### Step 1: Dataset Splitting

Generates CSV splits for 5 seeds and 4 strategies (random, cold_drug, cold_target, all_cold).

```bash
python split_all_dataset.py
```

Outputs: `split_data/seed_{seed}/{strategy}/{dataset}_{train|val|test}.csv`

#### Step 2: Protein Sequence Features

Extracts mean-pooled ESM-2 embeddings for each protein sequence.

```bash
python protein_process.py --dataset kiba --gpu_idx 1
```

Outputs: `data/kiba/preprocessed/sequence/{key}.npy`

#### Step 3: Protein Surface Features

Extracts dMaSIF surface point embeddings and maps them to residues with a chosen `k`.

```bash
python surface_process.py --dataset kiba --k 5
```

Outputs:

- `data/kiba/preprocessed/surface_points/{key}.pt`
- `data/kiba/preprocessed/residue_surface/k5/{key}.pt`

#### Step 4: Build Global Caches

Assembles drug graphs, fingerprints, protein graphs, and ESM features into cache files per `k`.

```bash
python create_data_motif.py --dataset kiba --surface_k 5 --gpu_idx 1
```

Outputs:

- `data/cache/kiba_smile_graph.pt`
- `data/cache/kiba_fingerprint.pt`
- `data/cache/kiba_protein_graphs_k5.pt`
- `data/cache/kiba_esm_feats.pt`

### Step 5: Training

```bash
# KIBA dataset, random split, seed 1, surface_k=5, GPU 1
python training_warmup.py \
  --dataset_idx 1 \
  --surface_k 5 \
  --gpu_idx 1 \
  --seed 1 \
  --strategy random \
  --lr 0.001 \
  --batch_size 512 \
  --epoch 500 \
  --patience 50
```

Key arguments:

- `--dataset_idx`: `0` for davis, `1` for kiba.
- `--surface_k`: must match the `k` used during preprocessing.
- `--gpu_idx`: target GPU device for training.
- `--strategy`: `random`, `cold_drug`, `cold_target`, or `all_cold`.
- `--seed`: random seed (must match the split seed).

Results and checkpoints are saved to:

```
results_{dataset}/{strategy}/seed_{seed}/
```

---

## Important Design Notes

### No Persistent Dataset `.pt` Files

`TestbedDatasetHMol` now inherits from `torch.utils.data.Dataset` instead of `PyG's InMemoryDataset`. It assembles data pairs **in-memory directly from global caches** on every run. There is no disk I/O for per-split `.pt` files. This eliminates redundant storage and makes switching between seeds/strategies instantaneous.

### Configurable Surface Mapping `k`

The nearest-neighbor parameter `k` for mapping dMaSIF surface points to residues is fully configurable (`3, 5, 8`, etc.). Each `k` produces an isolated cache (`protein_graphs_k{k}.pt`), so multiple experiments can coexist without collision.

### Explicit GPU Control for ESM Inference

Both `protein_process.py` and `create_data_motif.py` accept `--gpu_idx`, so ESM-2 inference can be directed to a specific GPU (e.g., `cuda:1`) rather than defaulting to `cuda:0`.

### Consolidated Preprocessing Paths

All protein preprocessing outputs are organized under a single directory:

```
data/{dataset}/preprocessed/
├── sequence/
├── surface_points/
└── residue_surface/k{k}/
```

---

## Model Hyperparameters

| Component       | Hyperparameter      | Value       |
| --------------- | ------------------- | ----------- |
| DrugMotifGAT    | `in_channels`     | 133         |
|                 | `hidden_channels` | 64          |
|                 | `out_channels`    | 128         |
|                 | `num_layers`      | 2           |
|                 | `heads`           | 2           |
|                 | `dropout`         | 0.2         |
| ProteinEGNN     | `num_features_xt` | 649         |
|                 | `hidden_nf`       | 128         |
|                 | `output_dim`      | 128         |
|                 | `n_layers`        | 4           |
| DualInteraction | `emb_dim`         | 128         |
|                 | `fp_dim`          | 1024        |
|                 | `esm_dim`         | 480         |
| MLP Decoder     | `hidden_dims`     | 1024 → 256 |

---

## Ablation Studies

Two additional training scripts are provided for ablation experiments:

- `training_warmup_ablation.py`: ablates fingerprint, ESM global, and multi-level representations.
- `training_warmup_ablation2.py`: ablates a2p attention, m2p attention, and protein max pooling.

---

## Citation

If you use HierDTA in your research, please cite:

```bibtex
@article{hierdta2024,
  title={HierDTA: Hierarchical Graph Neural Networks for Drug-Target Affinity Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

---

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
