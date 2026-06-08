import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.chemutils import build_himgnn_mol_hetero_dict


ECFP_DIM = 1024
CONTACT_THRESHOLD = 0.5


def load_pickle(path):
    with open(path, "rb") as pkl_file:
        return pickle.load(pkl_file)


def build_mol_hetero_dict(smiles):
    """Build a cache-friendly heterogeneous molecular graph."""
    return build_himgnn_mol_hetero_dict(smiles)


def target2graph(contact_map, protein_features, threshold=CONTACT_THRESHOLD):
    """Build the full-length protein graph from ESM2 contact probabilities.

    Self-loops and sequential-neighbour edges are FORCED to 1.0:
    - Without them, 300-node graphs at ESM2 threshold=0.5 have ~0.6% density
      (~2 edges/node), starving GAT of message-passing paths.
    - Self-loops prevent attention normalisation collapse.
    - Sequential edges create a backbone along the chain so information can
      propagate even where ESM2 predicts no long-range contact.
    """
    residue_features = protein_features[1:-1].astype(np.float32)
    target_size = residue_features.shape[0]
    contact_map = contact_map[:target_size, :target_size].copy()

    for i in range(target_size):
        contact_map[i, i] = 1.0
        if i + 1 < target_size:
            contact_map[i, i + 1] = 1.0

    src, dst = np.where(contact_map >= threshold)
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = contact_map[src, dst].astype(np.float32)
    print(
        f"  residue_features shape: {residue_features.shape}, "
        f"edge_index shape: {edge_index.shape}, "
        f"density: {edge_index.shape[1] / (target_size * (target_size - 1)) * 100:.1f}%"
    )
    return residue_features, edge_index, edge_weight


def load_surface_embedding(path):
    """Load a variable-length dMaSIF embedding consumed by the surface branch."""
    surface = torch.load(path, map_location="cpu", weights_only=False)
    embedding = surface["embedding"].float()
    if embedding.dim() != 2 or embedding.size(1) != 128:
        raise ValueError(f"Expected surface embedding [N, 128], got {tuple(embedding.shape)}")
    if embedding.size(0) == 0 or embedding.size(0) > 512:
        raise ValueError(f"Expected 1 to 512 surface points, got {embedding.size(0)}")
    return embedding


def main():
    parser = argparse.ArgumentParser(description="Build global feature caches for DTA datasets")
    parser.add_argument("--dataset", type=str, default="davis",
                        help="Dataset name to process")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_root = PROJECT_ROOT / "data" / dataset
    cache_dir = PROJECT_ROOT / "data" / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    drugs_df = pd.read_csv(dataset_root / f"{dataset}_drugs.csv")
    prots_df = pd.read_csv(dataset_root / f"{dataset}_prots.csv")
    esmc = load_pickle(dataset_root / f"{dataset}_esmc_pretrain.pkl")
    contact_maps = load_pickle(dataset_root / f"{dataset}_esm2_contact_map.pkl")

    print(f"Dataset: {dataset}")
    print(f"Cache directory: {cache_dir}")

    print("=" * 60)
    print("Phase 1: Building drug caches...")
    print("=" * 60)
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_DIM)
    drug_features = {}
    for smiles in set(drugs_df["compound_iso_smiles"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")
        graph = build_mol_hetero_dict(smiles)
        if graph is None:
            raise ValueError(f"Cannot build heterogeneous graph for SMILES: {smiles}")
        drug_features[smiles] = {
            "fingerprint": mg.GetFingerprintAsNumPy(mol),
            "hetero_graph": graph,
        }

    torch.save(drug_features, cache_dir / f"{dataset}_drug_features.pt")
    print(f"  Unique drugs: {len(drug_features)}")

    print("=" * 60)
    print("Phase 2: Building ESMC contact graphs and surface caches...")
    print("=" * 60)
    protein_features = {}
    target_keys = set(prots_df["target_key"].tolist())
    missing_keys = []

    for key in target_keys:
        if key not in esmc["mat_dict"] or key not in esmc["vec_dict"]:
            print(f"  ESMC feature not found. Skipping key {key}.")
            missing_keys.append(key)
            continue
        if key not in contact_maps["contact_map"]:
            print(f"  ESM2 contact map not found. Skipping key {key}.")
            missing_keys.append(key)
            continue

        surface_file = dataset_root / "surface_points" / f"{key}.pt"
        if not os.path.exists(surface_file):
            print(f"  Surface points not found: {surface_file}. Skipping key {key}.")
            missing_keys.append(key)
            continue

        node_features, edge_index, edge_weight = target2graph(
            contact_maps["contact_map"][key],
            esmc["mat_dict"][key],
        )
        protein_features[key] = {
            "x": node_features,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "surface_embedding": load_surface_embedding(surface_file),
            "esm_global": np.asarray(esmc["vec_dict"][key], dtype=np.float32),
        }

    if missing_keys:
        raise RuntimeError(
            f"Cannot build complete {dataset} protein caches: "
            f"{len(missing_keys)} target keys are missing required inputs. "
            "Run preprocessing/surface_process.py first and review the messages above."
        )

    torch.save(protein_features, cache_dir / f"{dataset}_protein_features.pt")
    print(f"  Unique proteins: {len(protein_features)}")
    print("Cache building complete.")


if __name__ == "__main__":
    main()
