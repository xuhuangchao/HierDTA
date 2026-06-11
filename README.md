# HiSurf-DTA

**HiSurf-DTA: Hierarchical Molecular and Surface-aware Protein Representation Learning for Drug-Target Affinity Prediction**

HiSurf-DTA is a multi-view drug-target affinity prediction framework. It combines hierarchical molecular graphs (atom + motif levels) with protein language-model-guided residue contact graphs and pocket surface features. The current implementation uses an **explicit cross-attention interaction module** (`Interaction`) that learns atom/motif-to-pocket-residue attention weights, replacing the earlier pure-concatenation baseline.

## Key Contributions

1. **Hierarchical molecular representation.** HiSurf-DTA jointly models atom-level graphs, motif-level graphs, and Morgan fingerprints (ECFP4 + MACCS + Topological) to capture local chemical environments, functional substructures, and global molecular topology. Atom and motif graphs are encoded independently by `GATv2Conv`-based `AtomGNN` branches.

2. **Language-model-guided protein graph encoding.** ESM-2 (3B) residue embeddings (480-dim) serve as node features for the pocket residue graph, and distance-based edges are constructed with a 10Å cutoff. An **E(n)-equivariant graph neural network (EGNN)** aggregates structural and sequence-aware protein information with coordinate-aware message passing, ensuring rotation/translation invariance of learned features.

3. **Independent pocket surface modeling.** Up to 512 dMaSIF surface point embeddings (128-dim) describe the local pocket environment. These are concatenated with ESM-2 embeddings as pocket node features (608-dim total, excluding 41-dim physical features), without requiring a potentially noisy residue-to-surface mapping.

4. **Explicit drug–protein cross-attention.** The `Interaction` module computes bidirectional attention — atom→pocket and motif→pocket — with a shared drug-side query projection. The two attention views are averaged to produce per-residue pocket weights, which are then used to pool pocket features into a single interaction context vector. This provides an interpretable binding-mode signal.

5. **Complementary global features.** Morgan fingerprint (3239-dim) and ESM-2 full-sequence mean-pool embedding (480-dim) are projected and concatenated with the interaction context, providing global molecular and protein-level information that complements the local cross-attention signal.

## Model Overview

The prediction head (`DTAPocketCrossHead`) receives three vectors after fusion:

1. **`interaction_ctx`** (128-dim): atom/motif-to-pocket cross-attention pooled pocket features. This captures which pocket residues are most relevant to the drug's atoms and functional motifs — the core interaction signal.

2. **`d_readout`** (128-dim): projected Morgan fingerprint (ECFP4 1024 + MACCS 166 + Topological 2048 = 3239 → 128). Captures global molecular topology and pharmacophore patterns.

3. **`p_readout`** (128-dim): projected ESM-2 full-sequence mean-pool embedding (480 → 128). Captures global protein sequence-level information from a pre-trained 3B-parameter protein language model.

The three 128-dim vectors are concatenated (384-dim) and passed to a 3-layer MLP regression head (384 → 1024 → 256 → 1).

### Architecture Diagram

```
SMILES                              PDB Pocket
   │                                     │
   ├─ HeteroGraph                       ├─ Pocket Graph
   │  ├─ atom nodes (37-dim)            │  ├─ ESM-2 emb (480-dim)
   │  ├─ motif nodes (50-dim)           │  ├─ dMaSIF surface (128-dim)
   │  └─ 3 edge types                   │  ├─ Cα coords (3-dim)
   │                                     │  └─ distance edges (2-dim)
   ▼                                     ▼
AtomGNN (GATv2Conv ×2)            PocketGraphEncoder (EGNN ×2)
   │                                     │
   ├─ atom_nodes [B,Na,128]              ├─ prot_nodes [B,Np,128]
   └─ motif_nodes [B,Nm,128]             │
         │         │                     │
         └────┬────┘                     │
              ▼                          ▼
         Interaction (Cross-Attention)
         ├─ atom → pocket attn
         ├─ motif → pocket attn
         └─ average → weighted pool
              │
              ▼
         interaction_ctx (128)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 [ctx]  [fp_proj]  [esm_proj]
 (128)   (128)      (128)
    │         │         │
    └─────────┼─────────┘
              ▼
         MLP (384→1024→256→1)
              │
              ▼
         Affinity Score
```

### Key Architectural Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Drug encoder | `GATv2Conv` (×2 layers) | Dynamic attention over atom/motif neighborhoods |
| Pocket encoder | `EGNN` (×2 layers, E(n)-equivariant) | Rotation/translation-invariant 3D structure encoding |
| Cross-attention | Single-head, shared query, dual-view average | Balances atom-level and motif-level binding signals |
| Fusion | [interaction_ctx ‖ fp_proj ‖ esm_proj] → MLP | Interaction signal + global drug info + global protein info |
| Embedding dim | 128 (all branches) | Uniform bottleneck for cross-modal compatibility |

## Data Layout

Place the source files under:

```text
data/{dataset}/
├── {dataset}_drugs.csv
├── {dataset}_prots.csv
├── {dataset}_esmc_pretrain.pkl
├── {dataset}_esm2_contact_map.pkl
└── pocket1_{dataset}/
```

The split files are expected at:

```text
data/{dataset}/seed_{seed}/{strategy}/
├── train.csv
├── valid.csv
└── test.csv
```

Supported split seeds:

```text
41, 42, 43, 32, 33
```

Supported strategies:

```text
warm, unseen_drug, unseen_prot, unseen_pair
```

## Preprocessing

Preprocessing scripts are grouped under:

```text
preprocessing/
├── esmc_pretrained.py      # ESM-C (600M) per-residue embeddings
├── esm2_map.py             # ESM-2 (3B) contact probability maps
├── surface_process.py      # dMaSIF pocket surface point extraction (≤512 points)
├── create_data.py          # Global drug & protein feature cache builder
├── create_drug_data.py     # Alternative drug cache (adds ChemBERTa token features)
├── cold_split.py           # Warm / cold-drug / cold-protein / cold-pair split generator
├── protein_process.py      # 41-dim physical residue features (type, distances, dihedrals)
├── build_pocket_graph.py   # Pocket residue graph: phys(41) + ESM2(480) + surface(128) = 649-dim
└── chemutils.py            # Heterogeneous molecular graph builder (atom + motif nodes)
```

If language-model features need to be regenerated, run:

```bash
python preprocessing/esmc_pretrained.py
python preprocessing/esm2_map.py
```

Run surface extraction before cache generation:

```bash
python preprocessing/surface_process.py --dataset davis
```

Then build the pocket graph cache:

```bash
python preprocessing/build_pocket_graph.py --dataset davis
```

Then build the global drug feature cache:

```bash
python preprocessing/create_data.py --dataset davis
```

Generate training splits with:

```bash
python preprocessing/cold_split.py --dataset davis --seeds 41 42 43 32 33
```

Generated caches:

```text
data/cache/{dataset}_drug_features.pt      # SMILES → {hetero_graph, fingerprint}
data/cache/{dataset}_pocket_graphs.pt      # target_key → {node_features[649], edge_index, coords, esm_global[480]}
```

During cache generation, `target2graph()` prints the residue feature shape and protein graph edge-index shape for each target.

## Training

Train one split:

```bash
python train.py --dataset davis --strategy warm --seed 41
```

Run all configured seeds for one strategy:

```bash
bash scripts/warm.sh
```

Equivalent scripts are available for `unseen_drug`, `unseen_prot`, and `unseen_pair`.

### Key Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 300 | Max training epochs |
| `--batch_size` | 256 | Batch size |
| `--lr` | 5e-4 | Learning rate (AdamW) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--patience` | 30 | Early stopping patience |
| `--hidden_dim` | 128 | Shared embedding dimension |
| `--dropout` | 0.2 | Dropout rate |
| `--atom_num_layers` | 2 | Atom GNN layers |
| `--motif_num_layers` | 2 | Motif GNN layers |
| `--pocket_num_layers` | 2 | Pocket EGNN layers |

The optimizer uses AdamW:

```python
torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

### Evaluation Metrics

Five metrics are computed on GPU:
- **RMSE** (Root Mean Squared Error)
- **MSE** (Mean Squared Error)
- **Pearson** correlation coefficient
- **CI** (Concordance Index)
- **RM²** (modified R²)
