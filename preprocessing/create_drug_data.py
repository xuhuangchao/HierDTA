import argparse
import os
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


ECFP_DIM = 2048
FINGERPRINT_DIM = ECFP_DIM


def build_mol_hetero_dict(smiles):
    """Build a cache-friendly heterogeneous molecular graph."""
    return build_himgnn_mol_hetero_dict(smiles)


def build_fingerprint(mol):
    """Build ECFP4 fingerprint."""
    ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_DIM)
    return ecfp_gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build drug feature caches for DTA datasets")
    parser.add_argument("--dataset", type=str, default="davis", help="Dataset name to process")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_root = PROJECT_ROOT / "data" / dataset
    cache_dir = PROJECT_ROOT / "data" / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    drugs_df = pd.read_csv(dataset_root / f"{dataset}_drugs.csv")

    print(f"Dataset: {dataset}")
    print(f"Cache directory: {cache_dir}")
    print(f"Fingerprint dim: {FINGERPRINT_DIM} (ECFP{ECFP_DIM})")

    drug_features = {}
    for smiles in set(drugs_df["compound_iso_smiles"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")

        graph = build_mol_hetero_dict(smiles)
        if graph is None:
            raise ValueError(f"Cannot build heterogeneous graph for SMILES: {smiles}")

        drug_features[smiles] = {
            "fingerprint": build_fingerprint(mol),
            "hetero_graph": graph,
        }

    torch.save(drug_features, cache_dir / f"{dataset}_drug_features.pt")
    print(f"  Unique drugs: {len(drug_features)}")
    print(f"  Fingerprint dim: {FINGERPRINT_DIM}")
    print("Drug cache building complete.")


if __name__ == "__main__":
    main()
