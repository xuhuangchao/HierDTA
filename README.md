# HiSurf-DTA

HiSurf-DTA is a drug-target affinity prediction model built from an atom-level molecular graph, a protein residue graph, an ECFP fingerprint, and a full-sequence ESM-2 protein embedding.

The current implementation is a compact pooled-fusion version:

- Drug graph: atom graph only, encoded by one `GATv2Conv` branch.
- Protein graph: residue graph from `{dataset}_protein_to_graph.pkl`, encoded by `GINConv`.
- Drug global feature: ECFP4 fingerprint, 2048 bits.
- Protein global feature: ESM-2 `esm2_t33_650M_UR50D` full-sequence mean embedding, 1280 dim.
- Fusion: concatenate graph/global vectors and predict affinity with fully connected layers.

## Current Architecture

### Drug Atom Graph

The molecular heterogeneous graph is still built by `preprocessing.chemutils`, but only the atom graph is consumed by the model.

```text
atom node feature: [N_atom, 37]
atom edge feature: [E_atom, 13]

GATv2Conv(37 -> 37, heads=2, concat=True)
ReLU
GraphPool(mean + add + max): [B, 37 * 2 * 3] = [B, 222]
Linear(222 -> 1024)
ReLU
Dropout(0.3)

drug_graph: [B, 1024]
```

The motif graph is currently not used.

### Protein Graph

Protein graph data is converted from:

```text
data/{dataset}/{dataset}_protein_to_graph.pkl
```

and saved as:

```text
data/cache/{dataset}_protein_graphs.pt
```

Each protein entry contains:

```python
{
    "node_features": Tensor[N, 41],
    "edge_index": Tensor[2, E],
    "edge_attr": Tensor[E, 10],
    "esm_global": Tensor[1280],
    "source_key": str,
}
```

The protein graph encoder uses:

```text
GINConv: 41 -> 256
ReLU
GraphPool(mean + add + max): [B, 256 * 3] = [B, 768]
Linear(768 -> 1024)
BatchNorm1d
ReLU
Dropout(0.3)

protein_graph: [B, 1024]
```

`edge_attr` is loaded and stored but is not consumed by the current `GINConv` encoder.

### Global Features

Drug fingerprint:

```text
ECFP4, radius=2, fpSize=2048
Linear(2048 -> 256)
BatchNorm1d
ReLU
Dropout(0.5)
```

Protein global feature:

```text
ESM-2 esm2_t33_650M_UR50D
full-sequence mean pooling
esm_global: [1280]

Linear(1280 -> 256)
BatchNorm1d
ReLU
Dropout(0.5)
```

Long protein sequences are processed in chunks of up to 1022 residues and then mean-pooled over all residue embeddings.

### Final Prediction Head

```text
concat:
  drug_graph      [B, 1024]
  protein_graph   [B, 1024]
  fp_proj         [B, 256]
  esm_global_proj [B, 256]

fusion input: [B, 2560]

Linear(2560 -> 2048) -> BatchNorm1d -> ReLU -> Dropout(0.5)
Linear(2048 -> 1024) -> BatchNorm1d -> ReLU -> Dropout(0.5)
Linear(1024 -> 512)  -> BatchNorm1d -> ReLU -> Dropout(0.5)
Linear(512 -> 1)
```

## Data Layout

Expected source files:

```text
data/{dataset}/
  {dataset}_drugs.csv
  {dataset}_prots.csv
  {dataset}_protein_to_graph.pkl
```

Expected split files:

```text
data/{dataset}/seed_{seed}/{strategy}/
  train.csv
  valid.csv
  test.csv
```

Supported split strategies:

```text
warm, unseen_drug, unseen_prot, unseen_pair
```

Supported seeds:

```text
41, 42, 43, 32, 33
```

## Preprocessing

Build drug features:

```bash
python preprocessing/create_drug_data.py --dataset davis
```

This creates:

```text
data/cache/{dataset}_drug_features.pt
```

with:

```python
{
    smiles: {
        "fingerprint": np.ndarray[2048],
        "hetero_graph": ...
    }
}
```

Build protein graph features:

```bash
python preprocessing/build_pocket_graph.py --dataset davis
```

This creates:

```text
data/cache/{dataset}_protein_graphs.pt
```

The script maps CSV target keys to legacy `{dataset}_protein_to_graph.pkl` keys. For DAVIS, the current resolver covers all targets:

```text
direct match: 404
fallback mapped: 38
missing: 0
```

Typical fallback cases include:

```text
ABL1(F317I)p -> ABL1(F317I)-phosphorylated
ABL1(F317I)  -> ABL1(F317I)-nonphosphorylated
RSK3(KinDom.1-N-terminal) -> RSK3(Kin.Dom.1-N-terminal)
EGFR(L747E749del) -> EGFR(L747-E749del, A750P)
```

For KIBA, the same resolver directly matches all CSV targets checked in this workspace:

```text
direct match: 228
fallback mapped: 0
missing: 0
```

`build_pocket_graph.py` requires `fair-esm` for ESM-2:

```bash
pip install fair-esm
```

Generate split files if needed:

```bash
python preprocessing/cold_split.py --dataset davis --seeds 41 42 43 32 33
```

## Training

Train one split:

```bash
python train.py --dataset davis --strategy warm --seed 41
```

Default training settings:

| Parameter | Default |
|-----------|---------|
| `--epochs` | 500 |
| `--batch_size` | 32 |
| `--lr` | 1e-4 |
| `--patience` | 30 |
| optimizer | Adam |

The optimizer is:

```python
torch.optim.Adam(model.parameters(), lr=args.lr)
```

## Choosing Graph Layer Counts

The current default is deliberately shallow:

```text
atom_num_layers = 1
pocket_num_layers = 1
```

This is a reasonable starting point for the current design because both branches use graph-level pooling and large dense fusion layers. The ECFP and ESM-2 global vectors already provide strong nonlocal information, so deeper message passing is not automatically better.

Recommended search:

| Branch | Try | Recommendation |
|--------|-----|----------------|
| Atom GATv2 | 1, 2, 3 | Start with 1. Try 2 if warm split underfits. Be cautious with 3. |
| Protein GIN | 1, 2, 3 | Start with 1 or 2. Use 2 if cold-protein improves. Avoid 3 unless validation supports it. |

Practical guidance:

- For DAVIS, atom graphs are small and atom features are already local; 1 GATv2 layer plus ECFP2048 is usually a strong baseline.
- For protein graphs, 1 GIN layer captures immediate residue-neighborhood signals; 2 layers may help propagate broader local structure.
- 3 layers can oversmooth protein node features and increase overfitting risk, especially with DAVIS-scale data.
- Because final fusion is large, evaluate deeper GNNs with validation MSE and cold-split metrics, not training loss.

Suggested ablation grid:

```text
atom_layers, protein_layers
1, 1
1, 2
2, 1
2, 2
```

Only try `(3, 2)` or `(2, 3)` after the four smaller settings show clear underfitting.

## Evaluation Metrics

Training reports:

- RMSE
- MSE
- Pearson
- CI
- rm2
