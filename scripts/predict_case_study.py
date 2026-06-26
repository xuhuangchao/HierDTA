"""Run a trained HierDTA checkpoint on a CSV drug library for one target.

The script builds and caches the same dual atom/motif drug representation used
during training, reuses the dataset protein cache, and writes a descending
KIBA-score ranking while preserving every input CSV column.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DTAModel
from models.dta_model import dta_collate_fn
from preprocessing.create_drug_data import build_fingerprint, build_mol_hetero_dict
from utils import TestbedDatasetHMol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict and rank a drug library with a trained HierDTA model"
    )
    parser.add_argument("--input", default="data/EGFR.csv")
    parser.add_argument("--smiles_column", default="drug_seq")
    parser.add_argument("--target", default="P00533")
    parser.add_argument("--dataset", default="kiba")
    parser.add_argument(
        "--checkpoint",
        default="results_kiba/warm/seed_41/ckpt_pool_dual_best.pt",
    )
    parser.add_argument(
        "--protein_cache", default="data/cache/kiba_protein_graphs.pt"
    )
    parser.add_argument(
        "--drug_cache", default="data/cache/egfr_case_study_drug_features.pt"
    )
    parser.add_argument(
        "--output", default="outputs/case_study/egfr_kiba_predictions.csv"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--rebuild_cache", action="store_true")
    return parser.parse_args()


def build_or_load_drug_features(smiles_values, cache_path, rebuild=False):
    cache_path = Path(cache_path)
    features = {}
    if cache_path.is_file() and not rebuild:
        features = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded drug cache: {cache_path} ({len(features)} structures)")

    missing = [smiles for smiles in smiles_values if smiles not in features]
    if not missing:
        return features

    print(f"Building features for {len(missing)} structures...")
    failures = []
    for index, smiles in enumerate(missing, start=1):
        if index == 1 or index % 250 == 0 or index == len(missing):
            print(f"  Drug features: {index}/{len(missing)}")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit could not parse SMILES")
            graph = build_mol_hetero_dict(smiles)
            if graph is None:
                raise ValueError("heterogeneous graph construction returned None")
            features[smiles] = {
                "fingerprint": build_fingerprint(mol),
                "hetero_graph": graph,
            }
        except Exception as exc:  # record the molecule and continue to a useful report
            failures.append({"smiles": smiles, "error": str(exc)})

    if failures:
        failure_path = cache_path.with_suffix(".failures.csv")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        raise RuntimeError(
            f"Feature construction failed for {len(failures)} structures; "
            f"details: {failure_path}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    print(f"Saved drug cache: {cache_path}")
    return features


def predict(model, loader, device):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            output = model(batch)
            predictions.append(output.detach().cpu().view(-1))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                print(f"  Inference batches: {batch_index}/{len(loader)}")
    return torch.cat(predictions).numpy()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = pd.read_csv(input_path)
    if args.smiles_column not in frame.columns:
        raise KeyError(
            f"Missing SMILES column '{args.smiles_column}'. Columns: {list(frame.columns)}"
        )
    if frame[args.smiles_column].isna().any():
        raise ValueError(f"Column '{args.smiles_column}' contains missing values")

    smiles = frame[args.smiles_column].astype(str).tolist()
    unique_smiles = list(dict.fromkeys(smiles))
    print(f"Input: {input_path}")
    print(f"Rows: {len(frame)}, unique SMILES: {len(unique_smiles)}")
    print(f"Target: {args.target}")

    drug_features = build_or_load_drug_features(
        unique_smiles, args.drug_cache, rebuild=args.rebuild_cache
    )
    protein_features = torch.load(
        args.protein_cache, map_location="cpu", weights_only=False
    )
    if args.target not in protein_features:
        raise KeyError(f"Target '{args.target}' is absent from {args.protein_cache}")

    dataset = TestbedDatasetHMol(
        xd=smiles,
        xt=[args.target] * len(smiles),
        y=[0.0] * len(smiles),
        dataset_name=args.dataset,
        drug_features=drug_features,
        pocket_features=protein_features,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dta_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    model = DTAModel(drug_graph_type="dual").to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint: {args.checkpoint}")

    scores = predict(model, loader, device)
    if len(scores) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} predictions, received {len(scores)}")

    result = frame.copy()
    result["hierdta_predicted_kiba_score"] = scores
    result = result.sort_values(
        "hierdta_predicted_kiba_score", ascending=False, kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "hierdta_rank", range(1, len(result) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    result.head(10).to_csv(output_path.with_name(f"{args.target}_kiba_top10.csv"), index=False)
    result.head(20).to_csv(output_path.with_name(f"{args.target}_kiba_top20.csv"), index=False)
    print(f"Saved full ranking: {output_path}")
    print(f"Score range: {scores.min():.6f} to {scores.max():.6f}")


if __name__ == "__main__":
    main()
