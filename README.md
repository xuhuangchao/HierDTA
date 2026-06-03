# HiSurf-DTA

**HiSurf-DTA: Hierarchical Molecular and Surface-aware Protein Representation Learning for Drug-Target Affinity Prediction**

HiSurf-DTA is a multi-view drug-target affinity prediction framework. It combines hierarchical molecular graphs with protein language-model-guided residue contact graphs and pocket surface features. The current implementation learns motif-residue interactions and fuses them with compact drug and target representations before affinity regression.

## Key Contributions

1. **Hierarchical molecular representation.** HiSurf-DTA jointly models atom-level graphs, motif-level graphs, and Morgan fingerprints to capture local chemical environments, functional substructures, and global molecular topology.
2. **Language-model-guided protein graph encoding.** ESM-C residue embeddings are used as node features, while ESM2 contact probabilities define weighted
   residue edges. A `GCNConv` and `GATConv` stack aggregates structural and sequence-aware protein information.
3. **Independent pocket surface modeling.** Up to 512 dMaSIF surface point embeddings describe the local pocket environment without requiring a potentially noisy residue-to-surface mapping. A mask excludes padded points when short pockets are batched.
4. **Complementary protein views.** The model combines the residue contact graph and the dMaSIF pocket surface representation.

## Model Overview

Atom and motif graphs are encoded by edge-aware `GATConv` stacks with
`BatchNorm1d`. Mean pooling produces graph-level molecular views, while
parameter-free atom-to-motif mean aggregation enriches motif tokens with fine-grained chemical
context.

A hidden-dimensional bilinear attention module learns motif-residue interaction weights. The
prediction head receives three compact vectors:

1. `local`: pooled motif-residue interaction representation.
2. `drug`: concatenated enriched motif and projected Morgan fingerprint views.
3. `target`: concatenated residue contact-graph and attention-pooled dMaSIF pocket views.

The global `drug-target` pair is the main prediction path. The motif-residue
interaction vector is projected and added through a lightweight local gate,
then a single MLP decoder predicts affinity.

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
├── esmc_pretrained.py
├── esm2_map.py
├── surface_process.py
├── create_data.py
├── cold_split.py
└── protein_process.py
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

Then build the two global cache files:

```bash
python preprocessing/create_data.py --dataset davis
```

Generate training splits with:

```bash
python preprocessing/cold_split.py --dataset davis --seeds 41 42 43 32 33
```

Generated caches:

```text
data/cache/{dataset}_drug_features.pt
data/cache/{dataset}_protein_features.pt
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

The optimizer is fixed-learning-rate Adam:

```python
torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
```
