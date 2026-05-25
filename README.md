# HierDTA: Hierarchical Drug-Target Affinity Prediction

HierDTA is a deep learning framework for **Drug-Target Affinity (DTA)** prediction that combines **AttentiveFP**-based molecular graph encoding with **E(n)-Equivariant Graph Neural Networks** for 3D protein pocket modeling. It leverages multi-modal protein representations — sequence embeddings (ESM-2), surface geometry (dMaSIF), and physicochemical residue features — integrated via a **bilinear co-attention** interaction module.

---

## 🏗️ Architecture Overview

HierDTA consists of three core components:

### 🧬 Drug Encoder — `DrugModel` / `AttentiveFP`

A **graph attention network** processes molecular graphs at the atom level.

- **Atom node features (266-dim)**:
  - **133-dim Chemprop features**: atomic number, degree, formal charge, chirality, hydrogen count, hybridization, aromaticity, mass — via one-hot encoding.
  - **133-dim ElementKG embeddings** (`ele2emb`): ontology-derived elemental knowledge embeddings.
- **Edge features (14-dim)**: bond type, ring membership, and other Chemprop-standard bond attributes.
- **Architecture**: `AttentiveFP` — GATEConv (attention-based message passing with GRU gating) followed by stacked `GATConv` layers with GRU skip connections. Default: 2 layers, hidden dim=64, output dim=128.
- **Output**: per-atom node features + **ECFP fingerprint** (1024-dim, computed via RDKit).

### 🔬 Protein Encoder — `ProteinEGNN` / `EGNN`

An **E(n)-Equivariant Graph Neural Network** encodes the 3D protein binding pocket with coordinate-awareness.

- **Residue node features (608-dim raw)**:
  - **41-dim physicochemical**: one-hot encoded residue type, hydrophobicity, polarity, charge, solvent accessibility, etc.
  - **480-dim ESM-2**: per-residue embeddings from `esm2_t12_35M_UR50D` (layer 12).
  - **128-dim surface geometry**: extracted by the **dMaSIF** encoder from protein surface point clouds, mapped to residues via KDTree (k nearest neighbors, configurable `k`).
- **Preprocessing**: The 41-dim physicochemical features are stripped before entering EGNN — only ESM-2 (480) + optionally surface (128) are fed to the equivariant layers.
- **Edges**: constructed with a distance cutoff between residue Cα coordinates (configurable).
- **Architecture**: 4-layer EGNN with SiLU activation, residual connections, and optional coordinate tanh gating. Default: hidden dim=128, output dim=128.

### 🔗 Interaction Module — `CoAttentionModule`

A **bidirectional cross-attention** mechanism with **bilinear fusion** models drug–protein interactions.

- **Drug → Protein cross-attention**: drug nodes attend to protein residues.
- **Protein → Drug cross-attention**: protein residues attend to drug nodes (shares attention scores for efficiency).
- **Global readout**:
  - Masked **mean pooling** on cross-attention context (interaction-aware).
  - **Max pooling** on raw encoder features (intrinsic signal).
  - Concatenated: `[mean_pool, max_pool]`.
- **Auxiliary branches** (ablative):
  - **Fingerprint branch**: ECFP (1024) → FC(512) → FC(128), fuses molecular fingerprint prior.
  - **Global ESM branch**: mean-pooled ESM-2 (480) → FC(256) → FC(128), fuses global protein prior.
- **Bilinear fusion**: enhanced drug representation × enhanced protein representation → 128-dim.
- **MLP decoder**: 128 → 256 → 1 (dropout=0.2).
- Output: predicted binding affinity.

---

## 📁 Directory Structure

```
HierDTA/
├── data/
│   ├── davis/                      # Davis dataset (raw)
│   │   ├── process.csv             #   Drug SMILES, target_key, affinity Y
│   │   ├── proteins.txt            #   Protein sequences
│   │   └── pocket1_davis/          #   Pocket PDB files
│   ├── kiba/                       # KIBA dataset (raw)
│   │   ├── process.csv
│   │   ├── proteins.txt
│   │   └── pocket1_kiba/
│   ├── cache/                      # Global feature caches (per dataset, per k)
│   │   ├── {dataset}_smile_graph.pt          # Drug atom graphs
│   │   ├── {dataset}_fingerprint.pt          # ECFP fingerprints
│   │   ├── {dataset}_protein_graphs_k{k}.pt  # Protein residue graphs
│   │   └── {dataset}_esm_feats.pt            # Global ESM-2 features
│   └── {dataset}/preprocessed/     # Intermediate preprocessing outputs
│       ├── sequence/               #   Per-protein ESM-2 embeddings (.npy)
│       ├── surface_points/         #   dMaSIF point cloud embeddings (.pt)
│       └── residue_surface/k{k}/   #   Surface→residue mapped features (.pt)
├── models/
│   ├── attentivefp.py              # AttentiveFP GNN implementation
│   ├── drug_model.py               # DrugModel wrapper
│   ├── egnn_clean.py               # E(n)-Equivariant GNN (EGCL layers)
│   ├── protein_model.py            # ProteinEGNN wrapper
│   └── model.py                    # HierDTA + CoAttentionModule + MLP decoder
├── dmasif_encoder/                 # dMaSIF surface geometry encoder
│   ├── protein_surface_encoder.py
│   └── geometry_processing.py
├── initial/
│   ├── ele2emb.pkl                 # ElementKG atom embeddings (133-dim)
│   ├── elementkgontology.embeddings.txt
│   ├── fg2emb.pkl                  # Functional group embeddings (legacy)
│   ├── rel2emb.pkl                 # Relation embeddings
│   ├── objectproperty.txt
│   └── get_dict.py
├── split_data/                     # Train/val/test CSV splits
│   └── seed_{seed}/{strategy}/{dataset}_{split}.csv
├── config.yml                      # dMaSIF model configuration
├── chemutils.py                    # Molecular graph construction utilities
├── utils.py                        # TestbedDatasetHMol + GPU metrics
├── split_all_dataset.py            # Step 1: dataset splitting
├── protein_process.py              # Step 2: ESM-2 sequence features
├── surface_process.py              # Step 3: dMaSIF surface extraction
├── create_data_motif.py            # Step 4: build global caches
├── run_preprocessing.py            # One-click preprocessing pipeline (Steps 1–4)
├── training_warmup.py              # Main training script
├── grid_search.py                  # Hyperparameter grid search
├── summarize_results.py            # Results aggregation (mean/std across seeds)
├── experiments.txt                 # Experiment plan & ablation checklist
└── environment.yml                 # Conda environment specification
```

---

## 📦 Installation

### Environment

Developed with **Python 3.10** and **PyTorch 2.5**. Conda is recommended:

```bash
conda env create -f environment.yml
conda activate pymesh
```

### Key Dependencies

| Package                                      | Purpose                                    |
| -------------------------------------------- | ------------------------------------------ |
| `torch >= 2.5`, `torch-geometric >= 2.6` | Deep learning framework + GNN              |
| `rdkit`                                    | Molecular graph construction, fingerprints |
| `MDAnalysis`                               | Protein structure parsing & analysis       |
| `fair-esm`                                 | ESM-2 protein language model               |
| `dmasif_encoder` (included)                | Protein surface geometry encoding          |
| `Biopython`                                | PDB parsing                                |
| `scipy`, `networkx`                      | Scientific computing, graph algorithms     |
| `pandas`, `numpy`                        | Data manipulation                          |

---

## 📊 Data Preparation

Place DTA datasets under `data/` with the following structure:

- **Davis**: `data/davis/process.csv`, `data/davis/proteins.txt`, `data/davis/pocket1_davis/*.pdb`
- **KIBA**: `data/kiba/process.csv`, `data/kiba/proteins.txt`, `data/kiba/pocket1_kiba/*.pdb`

`process.csv` must contain: `Drug` (SMILES), `target_key` (protein ID), `Y` (binding affinity).

---

## 🚀 Usage

### Option A: One-Click Preprocessing

```bash
python run_preprocessing.py \
  --dataset kiba \
  --k_list 3,5,8 \
  --gpu_idx 1
```

Runs **Steps 1–4** with smart skip logic — already-computed outputs are not reprocessed. Supports `--skip_split`, `--skip_protein`, `--skip_surface`, and `--force_cache` / `--force_surface` for fine-grained control.

### Option B: Step-by-Step

#### 1️⃣ Step 1 — Dataset Splitting

Generates train/val/test CSV splits across 5 seeds and 4 strategies.

```bash
python split_all_dataset.py
```

| Strategy        | Description                  |
| --------------- | ---------------------------- |
| `random`      | Random split (80/10/10)      |
| `cold_drug`   | Unseen drugs in test set     |
| `cold_target` | Unseen proteins in test set  |
| `all_cold`    | Both unseen drugs & proteins |

Output: `split_data/seed_{seed}/{strategy}/{dataset}_{train|val|test}.csv`

#### 2️⃣ Step 2 — Protein Sequence Features

Extracts mean-pooled ESM-2 (480-dim) for each protein.

```bash
python protein_process.py --dataset kiba --gpu_idx 1
```

Output: `data/kiba/preprocessed/sequence/{key}.npy`

#### 3️⃣ Step 3 — Surface Feature Extraction

Runs dMaSIF on pocket PDBs, maps surface point embeddings to residues via KDTree.

```bash
python surface_process.py --dataset kiba --k 5
```

Output:

- `data/kiba/preprocessed/surface_points/{key}.pt`
- `data/kiba/preprocessed/residue_surface/k5/{key}.pt`

#### 4️⃣ Step 4 — Build Global Caches

Assembles drug graphs, fingerprints, protein graphs, and ESM features into `.pt` cache files.

```bash
python create_data_motif.py --dataset kiba --surface_k 5 --gpu_idx 1
```

Output:

- `data/cache/kiba_smile_graph.pt`
- `data/cache/kiba_fingerprint.pt`
- `data/cache/kiba_protein_graphs_k5.pt`
- `data/cache/kiba_esm_feats.pt`

#### 5️⃣ Step 5 — Training

```bash
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

**Core arguments:**

| Argument          | Default    | Description                                              |
| ----------------- | ---------- | -------------------------------------------------------- |
| `--dataset_idx` | `0`      | `0` = Davis, `1` = KIBA                              |
| `--surface_k`   | `5`      | Must match preprocessing `k`                           |
| `--strategy`    | `random` | `random`, `cold_drug`, `cold_target`, `all_cold` |
| `--seed`        | `1`      | Must match split seed                                    |
| `--lr`          | `0.001`  | Learning rate                                            |
| `--batch_size`  | `512`    | Batch size                                               |
| `--epoch`       | `500`    | Max training epochs                                      |
| `--patience`    | `50`     | Early stopping patience                                  |

**Drug architecture:**

| Argument            | Default | Description                                |
| ------------------- | ------- | ------------------------------------------ |
| `--drug_hidden`   | `64`  | AttentiveFP hidden channels                |
| `--drug_out`      | `128` | Drug output dim (must equal `--emb_dim`) |
| `--n_layers_drug` | `2`   | Number of GNN layers                       |
| `--dropout`       | `0.2` | Dropout rate                               |

**Protein architecture:**

| Argument               | Default | Description                                   |
| ---------------------- | ------- | --------------------------------------------- |
| `--protein_hidden`   | `128` | EGNN hidden dim                               |
| `--protein_out`      | `128` | Protein output dim (must equal `--emb_dim`) |
| `--n_layers_protein` | `4`   | Number of EGNN layers                         |
| `--use_surface`      | `1`   | `1` = use dMaSIF surface, `0` = ESM-only  |

**Interaction & ablation:**

| Argument              | Default | Description                       |
| --------------------- | ------- | --------------------------------- |
| `--emb_dim`         | `128` | Interaction embedding dimension   |
| `--use_fingerprint` | `1`   | Include ECFP fingerprint branch   |
| `--use_p_global`    | `1`   | Include global ESM protein branch |

---

## 📈 Evaluation Metrics

All metrics are computed on GPU:

| Metric                  | Symbol | Description                |
| ----------------------- | ------ | -------------------------- |
| Root Mean Squared Error | RMSE   | Prediction error magnitude |
| Mean Squared Error      | MSE    | Squared prediction error   |
| Pearson Correlation     | R      | Linear correlation         |
| Concordance Index       | CI     | Ranking consistency        |
| RM²                    | RM²   | External validation metric |

Results saved to: `results_{dataset}/{strategy}/seed_{seed}/`

---

## 🔍 Hyperparameter Search

Run grid search over key hyperparameters:

```bash
python grid_search.py --dataset kiba --gpu_idx 0 --strategy random
```

The search grid (defined in `grid_search.py`) covers dropout, drug_hidden, learning rate, and weight_decay. Results are saved to `results_grid/{dataset}/{strategy}/`.

---

## 📋 Experiment Plans

See `experiments.txt` for a structured experiment checklist covering:

1. **Baseline** — random, cold_drug, cold_target, all_cold splits (5 seeds each)
2. **Ablation** — remove surface, fingerprint, or global ESM
3. **Architecture sweep** — deeper/wider encoders, more heads, different depths/dropouts
4. **Cold-start robustness** — cold_drug with ablated components
5. **Quick validation** — fast single-seed tuning runs

---

## 📊 Results Aggregation

Aggregate metrics across multiple seeds:

```bash
python summarize_results.py --base_dir results_kiba --strategy random
```

Computes mean ± std across seeds for all metrics.

---

## 🧪 Ablation Study Reference

| Experiment              | Flag                    | Purpose                              |
| ----------------------- | ----------------------- | ------------------------------------ |
| Remove surface features | `--use_surface 0`     | Test contribution of dMaSIF geometry |
| Remove fingerprint      | `--use_fingerprint 0` | Test ECFP prior necessity            |
| Remove global ESM       | `--use_p_global 0`    | Test global protein prior necessity  |

---

## 📝 Citation

If HierDTA aids your research, please cite:

```bibtex
@article{hierdta2024,
  title={HierDTA: Hierarchical Graph Neural Networks for Drug-Target Affinity Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 📬 Contact

For questions or issues, please open an issue on GitHub or contact the authors.
