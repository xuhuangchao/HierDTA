"""Build protein graph cache with full-sequence ESM-2 global embeddings.

Input:
  data/{dataset}/{dataset}_protein_to_graph.pkl
  data/{dataset}/{dataset}_prots.csv

Output:
  data/cache/{dataset}_protein_graphs.pt

Each saved target entry uses the target_key from {dataset}_prots.csv:
  node_features: [N, 41]
  edge_index:    [2, E]
  edge_attr:     [E, 10]
  esm_global:    [1280]
  source_key:    original key in {dataset}_protein_to_graph.pkl
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ESM_LAYER = 33
ESM_DIM = 1280
DEFAULT_CHUNK_SIZE = 1022


def resolve_protein_graph_key(key, protein_graphs):
    """Map DAVIS-style phospho suffixes to old protein_to_graph.pkl keys."""
    if key in protein_graphs:
        return key

    normalized = normalize_key(key)
    normalized_matches = [
        raw_key for raw_key in protein_graphs
        if normalize_key(raw_key) == normalized
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    prefix_matches = [
        raw_key for raw_key in protein_graphs
        if normalize_key(raw_key).startswith(normalized)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    if key.endswith("p"):
        candidate = f"{key[:-1]}-phosphorylated"
    else:
        candidate = f"{key}-nonphosphorylated"

    if candidate in protein_graphs:
        return candidate

    normalized_candidate = normalize_key(candidate)
    normalized_matches = [
        raw_key for raw_key in protein_graphs
        if normalize_key(raw_key) == normalized_candidate
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    prefix_matches = [
        raw_key for raw_key in protein_graphs
        if normalize_key(raw_key).startswith(normalized_candidate)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    raise KeyError(f"Protein graph not found for target key: {key}")


def normalize_key(key):
    return re.sub(r"[^a-z0-9]", "", key.lower())


def load_esm2_650m(device):
    import esm

    print("Loading ESM-2 model: esm2_t33_650M_UR50D")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    return model, alphabet.get_batch_converter()


@torch.no_grad()
def mean_pool_full_sequence(sequence, label, model, batch_converter, device, chunk_size):
    """Mean-pool ESM2 residue embeddings over the full sequence via chunks."""
    if not sequence:
        return torch.zeros(ESM_DIM, dtype=torch.float32)

    pooled_sum = torch.zeros(ESM_DIM, dtype=torch.float32)
    residue_count = 0

    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start:start + chunk_size]
        _, _, tokens = batch_converter([(label, chunk)])
        tokens = tokens.to(device)
        result = model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
        reps = result["representations"][ESM_LAYER][0, 1:len(chunk) + 1].detach().cpu()
        pooled_sum += reps.sum(dim=0)
        residue_count += reps.size(0)

        del result, reps, tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pooled_sum / max(residue_count, 1)


def build_protein_graphs(dataset_name, data_root=PROJECT_ROOT / "data", cache_dir=None, chunk_size=DEFAULT_CHUNK_SIZE):
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name
    cache_dir = Path(cache_dir) if cache_dir is not None else data_root / "cache"
    os.makedirs(cache_dir, exist_ok=True)

    prot_csv = dataset_dir / f"{dataset_name}_prots.csv"
    graph_pkl = dataset_dir / f"{dataset_name}_protein_to_graph.pkl"
    if not prot_csv.exists():
        raise FileNotFoundError(f"Protein CSV not found: {prot_csv}")
    if not graph_pkl.exists():
        raise FileNotFoundError(f"Protein graph pickle not found: {graph_pkl}")

    prots_df = pd.read_csv(prot_csv)
    with open(graph_pkl, "rb") as handle:
        raw_graphs = pickle.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, batch_converter = load_esm2_650m(device)

    protein_graphs = {}
    missing_keys = []
    for row in tqdm(prots_df.itertuples(index=False), total=len(prots_df), desc="Building protein graphs"):
        key = row.target_key
        sequence = row.target_sequence
        try:
            source_key = resolve_protein_graph_key(key, raw_graphs)
        except KeyError:
            missing_keys.append(key)
            continue

        node_features, edge_index, edge_attr = raw_graphs[source_key]
        protein_graphs[key] = {
            "node_features": torch.as_tensor(node_features, dtype=torch.float32),
            "edge_index": torch.as_tensor(edge_index, dtype=torch.int64).t().contiguous(),
            "edge_attr": torch.as_tensor(edge_attr, dtype=torch.float32),
            "esm_global": mean_pool_full_sequence(
                sequence=sequence,
                label=key,
                model=model,
                batch_converter=batch_converter,
                device=device,
                chunk_size=chunk_size,
            ),
            "source_key": source_key,
        }

    if missing_keys:
        raise RuntimeError(
            f"Missing {len(missing_keys)} protein graph keys. "
            f"Examples: {missing_keys[:10]}"
        )

    cache_path = cache_dir / f"{dataset_name}_protein_graphs.pt"
    torch.save(protein_graphs, cache_path)
    print(f"Saved {len(protein_graphs)} protein graphs to {cache_path}")
    return cache_path


def main():
    parser = argparse.ArgumentParser(description="Build protein graph cache with ESM2 global embeddings")
    parser.add_argument("--dataset", type=str, default="davis", help="Dataset name")
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "data" / "cache"))
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    build_protein_graphs(
        dataset_name=args.dataset,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
