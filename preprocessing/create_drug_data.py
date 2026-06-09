import argparse
import numpy as np
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


def build_mol_hetero_dict(smiles):
    """Build a cache-friendly heterogeneous molecular graph."""
    return build_himgnn_mol_hetero_dict(smiles)


def main():
    parser = argparse.ArgumentParser(description="Build drug feature caches for DTA datasets")
    parser.add_argument("--dataset", type=str, default="davis",
                        help="Dataset name to process")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_root = PROJECT_ROOT / "data" / dataset
    cache_dir = PROJECT_ROOT / "data" / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    drugs_df = pd.read_csv(dataset_root / f"{dataset}_drugs.csv")

    # Load ChemBERTa pretrained features
    chem_file = dataset_root / f"{dataset}_chem_pretrained.pkl"
    if chem_file.exists():
        with open(chem_file, "rb") as f:
            chem_data = pickle.load(f)
        mat_dict = chem_data.get("mat_dict", {})
        # Build SMILES → drug_key mapping (convert to str for mat_dict lookup)
        smiles_to_key = dict(zip(drugs_df["compound_iso_smiles"], drugs_df["drug_key"].astype(str)))
        print(f"ChemBERTa features loaded: {len(mat_dict)} drugs")
    else:
        mat_dict = {}
        smiles_to_key = {}
        print(f"WARNING: {chem_file} not found — ChemBERTa features skipped")

    print(f"Dataset: {dataset}")
    print(f"Cache directory: {cache_dir}")

    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_DIM)

    # ── First pass: collect ChemBERTa tokens to determine padding length ──
    print("Collecting ChemBERTa token lengths...")
    token_dict = {}
    token_lens = []
    for smiles in set(drugs_df["compound_iso_smiles"].tolist()):
        drug_key = smiles_to_key.get(smiles, "")
        if drug_key and drug_key in mat_dict:
            raw = mat_dict[drug_key].numpy() if hasattr(mat_dict[drug_key], 'numpy') else mat_dict[drug_key]
            token_dict[smiles] = raw
            token_lens.append(raw.shape[0])

    max_tokens = max(token_lens) if token_lens else 1
    print(f"  Found {len(token_dict)}/{len(set(drugs_df['compound_iso_smiles']))} drugs with ChemBERTa tokens")
    print(f"  Token lengths — min: {min(token_lens)}, max: {max_tokens}, mean: {np.mean(token_lens):.1f}")
    print(f"  Feature dim: {next(iter(token_dict.values())).shape[1]}")

    # ── Second pass：build drug cache with padded ChemBERTa ──
    drug_features = {}
    for smiles in set(drugs_df["compound_iso_smiles"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")
        graph = build_mol_hetero_dict(smiles)
        if graph is None:
            raise ValueError(f"Cannot build heterogeneous graph for SMILES: {smiles}")

        # ChemBERTa token features — padded to fixed length
        if smiles in token_dict:
            raw = token_dict[smiles]
            n = raw.shape[0]
            padded = np.zeros((max_tokens, raw.shape[1]), dtype=np.float32)
            padded[:n] = raw
            mask = np.zeros(max_tokens, dtype=np.bool_)
            mask[:n] = True
        else:
            padded = np.zeros((max_tokens, 384), dtype=np.float32)
            mask = np.zeros(max_tokens, dtype=np.bool_)

        drug_features[smiles] = {
            "fingerprint": mg.GetFingerprintAsNumPy(mol),
            "hetero_graph": graph,
            "chemberta_tokens": padded,    # [max_tokens, 384]
            "chemberta_mask": mask,         # [max_tokens]
        }

    torch.save(drug_features, cache_dir / f"{dataset}_drug_features.pt")
    print(f"  Unique drugs: {len(drug_features)}")
    print(f"  chemberta_tokens shape: [max_tokens={max_tokens}, 384]")
    print("Drug cache building complete.")


if __name__ == "__main__":
    main()
