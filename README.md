# HierDTA

Hierarchical Drug-Target Affinity prediction model that jointly encodes atom-level and motif-level molecular graphs, a protein residue graph, and global sequence/fingerprint features.

## 🏗️ Architecture Overview

```
                    ┌──────────────────────────────┐
                    │       Drug Encoder(s)         │
                    │                               │
  data.hetero ──────┤  atom:  GATv2Conv on [N, 37] │──► drug_graph [B, 1024]
                    │  motif: GATv2Conv on [M, 50]  │    (or dual → gated fusion)
                    └──────────────────────────────┘
                    ┌──────────────────────────────┐
  data.protein_graph ┤ GINConv on [N_res, 41]       │──► protein_graph [B, 1024]
  data.esm_global ──┤  Linear(1280 → 256)           │──► esm_proj  [B,  256]
  data.fingerprint ─┤  Linear(2048 → 256)           │──► fp_proj   [B,  256]
                    └──────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    FusionHead       │
                    │  cat → MLP → 1      │
                    └────────────────────┘
```

## 💊 Drug Graph Encoder

The molecular heterogeneous graph is built by `preprocessing/chemutils.py` via HimGNN-style functional-group decomposition (38 rule-based detectors from `func_group/Substructure_Extraction.py`). The graph contains two node types and three edge types:

| Component          | Key                                  | Shape                          | Description                                                                      |
| ------------------ | ------------------------------------ | ------------------------------ | -------------------------------------------------------------------------------- |
| Atom nodes         | `x_atom`                           | `[N_atom, 37]`               | One-hot element type, valence, H-count, hybridization, aromatic, ring, chirality |
| Bond edges         | `aa_edge_index` / `aa_edge_attr` | `[2, E_aa]` / `[E_aa, 13]` | Bond type, in-ring, conjugated, stereo                                           |
| Motif nodes        | `x_motif`                          | `[N_motif, 50]`              | 37-dim atom-feature sum + 13-dim internal bond-feature sum per functional group  |
| Atom→Motif edges  | `am_edge_index` / `am_edge_attr` | `[2, E_am]` / `[E_am, 1]`  | Membership (constant 1.0)                                                        |
| Motif→Motif edges | `mm_edge_index` / `mm_edge_attr` | `[2, E_mm]` / `[E_mm, 37]` | Shared-atom or inter-motif bond; edge attr = summed atom features                |

### Drug Graph Modes (`--drug_graph_type`)

| Mode      | Encoders                     | Fusion          | Description                                                             |
| --------- | ---------------------------- | --------------- | ----------------------------------------------------------------------- |
| `atom`  | 1× GATv2Conv on atom nodes  | —              | Baseline. Atom-level graph only.                                        |
| `motif` | 1× GATv2Conv on motif nodes | —              | Motif-level graph only. Functional-group semantics.                     |
| `dual`  | 2× GATv2Conv (atom + motif) | GatedDrugFusion | Per-dimension sigmoid gate soft-selects between atom and motif signals. |

#### Atom Branch

```text
GATv2Conv(37 → 37, heads=2, concat=True, edge_dim=13)
ReLU
GraphPool(mean + add + max): [B, 37×2×3] = [B, 222]
Linear(222 → 1024) → ReLU → Dropout(0.3)
drug_atom: [B, 1024]
```

#### Motif Branch

```text
GATv2Conv(50 → 50, heads=2, concat=True, edge_dim=37)
ReLU
GraphPool(mean + add + max): [B, 50×2×3] = [B, 300]
Linear(300 → 1024) → ReLU → Dropout(0.3)
drug_motif: [B, 1024]
```

#### Gated Drug Fusion (dual mode only)

```text
atom_vec [B, 1024] ──► atom_proj [B, 1024] ──┐
                                               ├── g * atom_h + (1-g) * motif_h → [B, 1024]
motif_vec [B, 1024] ─► motif_proj [B, 1024] ─┘
                              ▲
gate = σ(Linear(cat(atom_vec, motif_vec))) [B, 1024]
```

The gate is a per-dimension sigmoid mask learned from the concatenated representations. `g ≈ 1` means the dimension relies on atom signal; `g ≈ 0` means it relies on motif signal. The fused output retains 1024 dimensions so the downstream `FusionHead` is unchanged.

#### Optional Bottom-Up Atom→Motif Update

With `--drug_graph_type dual --atom_motif_mode bottom_up`, atom embeddings
produced by the atom GATv2 branch are mean-aggregated along the explicit
`atom-in-motif` membership edges. A gated residual update injects the resulting
context into the original 50-dimensional motif features before motif-level
message passing. The default `--atom_motif_mode none` preserves the independent
dual-branch behavior.

## 🧬 Protein Graph Encoder

Residue-level protein graph derived from kinase-related domain(s) in `data/{dataset}/{dataset}_protein_to_graph.pkl`, cached as `data/cache/{dataset}_protein_graphs.pt`.

Each protein entry:

```python
{
    "node_features": Tensor[N_res, 41],   # residue node features
    "edge_index":    Tensor[2, E],         # interaction edges within cutoff
    "edge_attr":     Tensor[E, 10],        # interaction attr (not used)
    "esm_global":    Tensor[1280],         # ESM-2 full-sequence mean embedding
    "source_key":    str,
}
```

Encoder:

```text
GINConv: 41 → 256
ReLU
GraphPool(mean + add + max): [B, 256×3] = [B, 768]
Linear(768 → 1024) → BatchNorm1d → ReLU → Dropout(0.3)
prot_graph: [B, 1024]
```

## 🌐 Global Features

| Feature           | Source                                    | Dim       | Projection                                      |
| ----------------- | ----------------------------------------- | --------- | ----------------------------------------------- |
| Drug fingerprint  | ECFP4, radius=2, fpSize=2048              | [B, 2048] | Linear(2048→256) → BN → ReLU → Dropout(0.5) |
| Protein embedding | ESM-2 `esm2_t33_650M_UR50D` mean-pooled | [B, 1280] | Linear(1280→256) → BN → ReLU → Dropout(0.5) |

Long protein sequences are chunked at 1022 residues; chunk embeddings are mean-pooled into the final 1280-dim vector.

## ⚡ Fusion Head

```text
concat([drug_graph, prot_graph, fp_proj, esm_proj]): [B, 2560]

Linear(2560 → 2048) → BatchNorm1d → ReLU → Dropout(0.5)
Linear(2048 → 1024) → BatchNorm1d → ReLU → Dropout(0.5)
Linear(1024 →  512) → BatchNorm1d → ReLU → Dropout(0.5)
Linear(512  →    1)
```

## 📂 Data Layout

### Source files

```text
data/{dataset}/
  {dataset}_drugs.csv
  {dataset}_prots.csv
  {dataset}_protein_to_graph.pkl
```

### Split files

```text
data/{dataset}/seed_{seed}/{strategy}/
  train.csv
  valid.csv
  test.csv
```

### Split strategies

| Strategy        | Description                                             |
| --------------- | ------------------------------------------------------- |
| `warm`        | Random split (80/10/10)                                 |
| `unseen_drug` | Cold-start: test drugs absent from training             |
| `unseen_prot` | Cold-start: test targets absent from training           |
| `unseen_pair` | Cold-start: test drug-target pairs absent from training |

### Seeds

`41, 42, 43, 32, 33`

## ⚙️ Preprocessing

### 1. Build drug features

```bash
python preprocessing/create_drug_data.py --dataset davis
```

Creates `data/cache/{dataset}_drug_features.pt`:

```python
{
    smiles: {
        "fingerprint": np.ndarray[2048],   # ECFP4
        "hetero_graph": {                   # full heterogeneous graph
            "x_atom":         np.ndarray[N, 37],
            "x_motif":        np.ndarray[M, 50],
            "aa_edge_index":  np.ndarray[2, E_aa],
            "aa_edge_attr":   np.ndarray[E_aa, 13],
            "am_edge_index":  np.ndarray[2, E_am],
            "am_edge_attr":   np.ndarray[E_am, 1],
            "mm_edge_index":  np.ndarray[2, E_mm],
            "mm_edge_attr":   np.ndarray[E_mm, 37],
        }
    }
}
```

### 2. Build protein graph features

Requires `fair-esm`:

```bash
pip install fair-esm
```

```bash
python preprocessing/build_protein_graph.py --dataset davis
```

Creates `data/cache/{dataset}_protein_graphs.pt`. The script maps CSV target keys to legacy `.pkl` keys with fallback resolution (e.g., `ABL1(F317I)p` → `ABL1(F317I)-phosphorylated`).

### 3. Generate split files

```bash
python preprocessing/cold_split.py --dataset davis --seeds 41 42 43 32 33
```

## 🚀 Training

Single run:

```bash
python train.py --dataset davis --strategy warm --seed 41 --drug_graph_type atom
```

Gated dual-fusion run:

```bash
python train.py --dataset davis --strategy unseen_drug --seed 41 --drug_graph_type dual
```

Batch training via scripts:

```bash
bash scripts/warm.sh
bash scripts/unseen_drug.sh
bash scripts/unseen_prot.sh
bash scripts/unseen_pair.sh
```

### Hyperparameters

| Parameter             | Default  | Description                      |
| --------------------- | -------- | -------------------------------- |
| `--epochs`          | 500      | Max training epochs              |
| `--batch_size`      | 32       | Batch size                       |
| `--lr`              | 1e-4     | Learning rate (Adam)             |
| `--patience`        | 30       | Early stopping patience          |
| `--drug_graph_type` | `atom` | `atom`, `motif`, or `dual` |
| `--atom_motif_mode` | `none` | `none` or `bottom_up`; `bottom_up` requires `dual` |

Optimizer: `torch.optim.Adam(model.parameters(), lr=args.lr)`. Loss: `nn.MSELoss`.

## 📊 Evaluation Metrics

Reported per epoch and at test time:

| Metric  | Description                     |
| ------- | ------------------------------- |
| RMSE    | Root Mean Squared Error         |
| MSE     | Mean Squared Error              |
| Pearson | Pearson correlation coefficient |
| CI      | Concordance Index               |
| r²ₘ   | Modified R² (rm²)             |

All metrics are computed on GPU via `utils.py`.

## 📁 Project Structure

```
HierDTA/
├── train.py                              # Training script
├── utils.py                              # Dataset (TestbedDatasetHMol), metrics, cache loading
├── .gitignore
│
├── models/
│   ├── __init__.py                       # Module exports
│   ├── dta_model.py                      # DTAModel, FusionHead, GatedDrugFusion, DTABatch, collate
│   └── encoder.py                        # AtomGNN (GATv2Conv), ProteinGraphEncoder (GINConv)
│
├── preprocessing/
│   ├── create_drug_data.py               # Drug feature cache builder (ECFP4 + hetero graph)
│   ├── build_protein_graph.py            # Protein graph cache builder (ESM-2 global embedding)
│   ├── chemutils.py                      # HimGNN heterogeneous molecular graph construction
│   ├── cold_split.py                     # Train/valid/test split generation (4 strategies)
│   ├── download_metz_alphafold.py        # AlphaFold structure downloader (Metz dataset)
│   └── build_metz_protein_to_graph.py    # Metz-specific protein graph construction
│
├── func_group/
│   ├── Substructure_Extraction.py        # 38 rule-based functional-group detectors
│   └── MolGraph_Construction.py          # Legacy DGL-based graph builder (not used)
│
├── scripts/
│   ├── warm.sh                           # python train.py --strategy warm
│   ├── unseen_drug.sh                    # python train.py --strategy unseen_drug
│   ├── unseen_prot.sh                    # python train.py --strategy unseen_prot
│   ├── unseen_pair.sh                    # python train.py --strategy unseen_pair
│   └── summarize_seed_results.py         # Aggregate metrics across seeds
│
├── data/
│   ├── davis/                            # DAVIS: drugs.csv, prots.csv, data.csv, PDBs
│   ├── kiba/                             # KIBA: drugs.csv, prots.csv, data.csv, PDBs
│   ├── metz/                             # Metz: drugs.csv, prots.csv, data.csv, PDBs, splits
│   ├── descriptors/                      # aa_phy7.txt, BLOSUM62_dim23.txt
│   └── cache/                            # Generated .pt feature caches 
│
└── results_{dataset}/                    # Checkpoints and per-seed metrics 
```

## 📝 Citation

If you use HierDTA in your research, please cite our work.

## 🙏 Acknowledgements

- Residue-level protein graph construction method adapted from [vtarasv/3d-prot-dta](https://github.com/vtarasv/3d-prot-dta)
- Functional-group motif decomposition adapted from [UnHans/HimGNN](https://github.com/UnHans/HimGNN)
