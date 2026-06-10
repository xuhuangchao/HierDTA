# -*- coding: utf-8 -*-
"""Build pocket residue graphs with ESM-2 + physical + surface features.

Preprocessing script. Run once per dataset before training.

Requires: pip install fair-esm

Output: data/cache/{dataset}_pocket_graphs.pt
  Per target key:
    node_features:   [N_res, 649]  (physical 41 + ESM2 480 + surface 128)
    residue_coords:  [N_res, 3]    (Cα coordinates for EGNN)
    edge_index:      [2, E]
    edge_weight:     [E, 2]        (min_dist, max_dist)
    esm_global:      [480]         (full-sequence ESM-2 mean pool)
"""

import argparse
import glob
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- Biopython PDB parsing ---
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import esm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────
#  Physical feature computation (41-dim per residue)
# ──────────────────────────────────────────────────────────

METAL = [
    "LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT",
    "ZN", "CO", "PD", "AG", "CR", "FE", "V", "MN", "HG", "GA", "CD", "YB",
    "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y",
    "LA", "CE", "PR", "ND", "GD", "TB", "DY", "ER", "TM", "LU", "HF", "ZR",
    "CE", "U", "PU", "TH",
]

RESIDUE_TYPES = [
    'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP',
    'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS',
    'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP',
    'MLY', 'PCA', 'LLP', 'M', 'X',
]

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "CSO": "C", "PTR": "Y", "TPO": "T", "SEP": "S",
    "KCX": "K", "CSD": "C", "MLY": "K", "PCA": "E", "LLP": "K",
}


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_resname(resname_str):
    if resname_str[:2] == "CA":
        return "CA"
    elif resname_str[:2] == "FE":
        return "FE"
    elif resname_str[:2] == "CU":
        return "CU"
    elif resname_str.strip() in METAL:
        return "M"
    return resname_str.strip()


def residue_to_aa(residue):
    return AA3_TO_1.get(obtain_resname(residue.get_resname()), "X")


def calc_self_dist(residue):
    try:
        atoms = list(residue.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])
        if len(coords) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        from scipy.spatial.distance import pdist
        dists = pdist(coords)
        max_d = dists.max() if len(dists) > 0 else 0.0
        min_d = dists.min() if len(dists) > 0 else 0.0
        ca = [a for a in atoms if a.get_name() == "CA"]
        c  = [a for a in atoms if a.get_name() == "C"]
        n  = [a for a in atoms if a.get_name() == "N"]
        o  = [a for a in atoms if a.get_name() == "O"]
        def d(a_list, b_list):
            if a_list and b_list:
                return np.linalg.norm(a_list[0].get_coord() - b_list[0].get_coord())
            return 0.0
        return [max_d * 0.1, min_d * 0.1, d(ca, o) * 0.1, d(o, n) * 0.1, d(n, c) * 0.1]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def calc_dihedral_angles(residue, prev_residue, next_residue):
    try:
        def v(res, name):
            atoms = [a for a in res.get_atoms() if a.get_name() == name]
            return atoms[0].get_vector() if atoms else None
        phi = 0.0
        if prev_residue:
            c_prev = v(prev_residue, "C")
            n_curr = v(residue, "N")
            ca_curr = v(residue, "CA")
            c_curr = v(residue, "C")
            if all([c_prev, n_curr, ca_curr, c_curr]):
                phi = calc_dihedral(c_prev, n_curr, ca_curr, c_curr) * 0.01
        psi = 0.0
        if next_residue:
            n_curr = v(residue, "N")
            ca_curr = v(residue, "CA")
            c_curr = v(residue, "C")
            n_next = v(next_residue, "N")
            if all([n_curr, ca_curr, c_curr, n_next]):
                psi = calc_dihedral(n_curr, ca_curr, c_curr, n_next) * 0.01
        return [phi, psi, 0.0, 0.0]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def calc_res_features(residue, all_residues_dict, all_residues_keys):
    resname = obtain_resname(residue.get_resname())
    ohe = one_of_k_encoding_unk(resname, RESIDUE_TYPES)
    sd = calc_self_dist(residue)
    current_idx = next(
        (idx for idx, rid in enumerate(all_residues_keys) if all_residues_dict[rid] is residue), -1)
    prev_idx = current_idx - 1
    next_idx = current_idx + 1
    prev_res = all_residues_dict[all_residues_keys[prev_idx]] if current_idx >= 0 and prev_idx >= 0 else None
    next_res = all_residues_dict[all_residues_keys[next_idx]] if 0 <= next_idx < len(all_residues_keys) else None
    da = calc_dihedral_angles(residue, prev_res, next_res)
    return np.array(ohe + sd + da, dtype=np.float32)


# ──────────────────────────────────────────────────────────
#  Edge construction (distance-based)
# ──────────────────────────────────────────────────────────

def build_edges(all_residues_dict, cutoff=10.0):
    residues = list(all_residues_dict.values())
    n = len(residues)
    edge_src, edge_dst, edge_min, edge_max = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            coords_i = np.array([a.get_coord() for a in residues[i].get_atoms()])
            coords_j = np.array([a.get_coord() for a in residues[j].get_atoms()])
            from scipy.spatial.distance import cdist
            dists = cdist(coords_i, coords_j)
            min_d = dists.min()
            if min_d <= cutoff:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_min.extend([min_d * 0.1, min_d * 0.1])
                edge_max.extend([dists.max() * 0.1, dists.max() * 0.1])
    if not edge_src:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 2, dtype=torch.float32))
    return (torch.tensor([edge_src, edge_dst], dtype=torch.long),
            torch.tensor(list(zip(edge_min, edge_max)), dtype=torch.float32))


# ──────────────────────────────────────────────────────────
#  Surface point matching
# ──────────────────────────────────────────────────────────

def match_surface_to_residues(residue_coords, surface_xyz, surface_emb, k=5):
    from scipy.spatial.distance import cdist
    ca_coords = np.array(residue_coords)
    dists = cdist(ca_coords, surface_xyz)
    nearest_idx = np.argpartition(dists, k, axis=1)[:, :k]
    surface_feats = []
    for res_i in range(len(residue_coords)):
        nearest_embs = surface_emb[nearest_idx[res_i]]
        surface_feats.append(nearest_embs.mean(dim=0))
    return torch.stack(surface_feats)


# ──────────────────────────────────────────────────────────
#  Pocket file lookup
# ──────────────────────────────────────────────────────────

def get_pocket_file(pocket_dir, key, suffix=".pdb"):
    processed_key = re.sub(r'[.\-() ]', '', key.lower())
    all_files = glob.glob(os.path.join(pocket_dir, f"{processed_key}*{suffix}"))
    return all_files[0] if all_files else None


# ──────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────

def build_pocket_graphs(dataset_name, data_root=PROJECT_ROOT / "data",
                        cache_dir=None, max_seq_len=1200):
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name
    pocket_dir = dataset_dir / f"pocket1_{dataset_name}"
    surface_dir = dataset_dir / "surface_points"

    if cache_dir is None:
        cache_dir = data_root / "cache"
    else:
        cache_dir = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    prot_csv = dataset_dir / f"{dataset_name}_prots.csv"
    if not prot_csv.exists():
        raise FileNotFoundError(f"Protein CSV not found: {prot_csv}")
    prot_df = pd.read_csv(prot_csv)
    prot_dict = dict(zip(prot_df["target_key"], prot_df["target_sequence"]))
    all_keys = list(prot_dict.keys())
    print(f"Dataset: {dataset_name}, proteins: {len(all_keys)}")

    print("Loading ESM-2 (esm2_t12_35M_UR50D)...")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 model loaded successfully")

    pocket_graphs = {}
    parser = PDBParser(QUIET=True)

    for key in tqdm(all_keys, desc="Building pocket graphs"):
        try:
            pocket_file = get_pocket_file(str(pocket_dir), key)
            if not pocket_file:
                print(f"  WARNING: No pocket PDB for {key}")
                continue

            structure = parser.get_structure(key, pocket_file)
            all_residues_dict = {}
            all_residues_keys = []
            for pdb_model in structure:
                for chain in pdb_model:
                    for res in chain:
                        if res.get_id()[0] == " ":
                            res_id = res.get_id()[1]
                            if res_id not in all_residues_keys:
                                all_residues_keys.append(res_id)
                                all_residues_dict[res_id] = res

            if not all_residues_dict:
                print(f"  WARNING: No valid residues in {pocket_file}")
                continue

            if len(all_residues_keys) > max_seq_len:
                print(f"  WARNING: Pocket residues for {key} exceed max_seq_len; truncating to {max_seq_len}")
                all_residues_keys = all_residues_keys[:max_seq_len]
                all_residues_dict = {rid: all_residues_dict[rid] for rid in all_residues_keys}

            # Physical features (41-dim)
            phys_feats = np.array([
                calc_res_features(all_residues_dict[rid], all_residues_dict, all_residues_keys)
                for rid in all_residues_keys
            ], dtype=np.float32)

            # Full-sequence ESM-2 → esm_global (480-dim)
            full_seq = prot_dict.get(key, "")
            if not full_seq:
                print(f"  WARNING: No sequence for {key}")
                continue
            truncated_seq = full_seq[:max_seq_len]
            batch_labels, batch_strs, batch_tokens = batch_converter([(key, truncated_seq)])
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                full_results = model(batch_tokens, repr_layers=[12])
            esm_full = full_results["representations"][12][0, 1:len(truncated_seq)+1].cpu()
            esm_global = esm_full.mean(dim=0)

            # Pocket sequence → ESM-2 per-residue (480-dim)
            pocket_seq = "".join(residue_to_aa(all_residues_dict[rid]) for rid in all_residues_keys)
            if not pocket_seq:
                print(f"  WARNING: Empty pocket sequence for {key}")
                continue
            _, _, pocket_tokens = batch_converter([(key, pocket_seq)])
            pocket_tokens = pocket_tokens.to(device)
            with torch.no_grad():
                pocket_results = model(pocket_tokens, repr_layers=[12])
            pocket_esm = pocket_results["representations"][12][0, 1:len(pocket_seq)+1].cpu()
            valid_res_ids = all_residues_keys
            valid_res_dict = all_residues_dict

            # Surface point matching & Cα coordinates
            surface_file = surface_dir / f"{key}.pt"
            surface_feat = torch.zeros(len(valid_res_ids), 128)

            ca_coords = []
            for rid in valid_res_ids:
                res = all_residues_dict[rid]
                ca_atoms = [a for a in res.get_atoms() if a.get_name() == "CA"]
                if ca_atoms:
                    ca_coords.append(ca_atoms[0].get_coord().tolist())
                else:
                    ca_coords.append([0.0, 0.0, 0.0])
            residue_coords = torch.tensor(ca_coords, dtype=torch.float32)

            if surface_file.exists():
                surf_data = torch.load(surface_file, map_location="cpu", weights_only=True)
                surface_xyz = surf_data["xyz"].numpy()
                surface_emb = surf_data["embedding"]
                ca_tuples = [tuple(c) for c in ca_coords]
                surface_feat = match_surface_to_residues(ca_tuples, surface_xyz, surface_emb, k=5)
            else:
                print(f"  WARNING: No surface file for {key}, surface features set to 0")

            # Node features = phys(41) + ESM2(480) + surface(128) = 649
            node_features = torch.cat([
                torch.from_numpy(phys_feats),
                pocket_esm,
                surface_feat,
            ], dim=-1)

            edge_index, edge_weight = build_edges(valid_res_dict)

            # Padded full-sequence ESM2
            L = esm_full.size(0)
            esm_full_padded = torch.zeros(max_seq_len, 480, dtype=torch.float32)
            esm_full_padded[:L] = esm_full
            esm_full_mask = torch.zeros(max_seq_len, dtype=torch.bool)
            esm_full_mask[:L] = True

            print(f"  {key}: node_features={list(node_features.shape)}, "
                  f"residue_coords={list(residue_coords.shape)}, "
                  f"edge_index={list(edge_index.shape)}")
            pocket_graphs[key] = {
                "node_features": node_features,
                "residue_coords": residue_coords,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "esm_global": esm_global,
                "esm_full": esm_full_padded,
                "esm_full_mask": esm_full_mask,
                "seq_len": len(truncated_seq),
            }

            del full_results, pocket_results, batch_tokens, pocket_tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR processing {key}: {e}")
            traceback.print_exc()
            continue

    cache_path = cache_dir / f"{dataset_name}_pocket_graphs.pt"
    torch.save(pocket_graphs, cache_path)
    print(f"\nSaved {len(pocket_graphs)} pocket graphs to {cache_path}")
    return cache_path


def main():
    parser = argparse.ArgumentParser(description="Build pocket residue graphs")
    parser.add_argument("--dataset", type=str, default="davis", help="Dataset name")
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--cache_dir", type=str, default=str(PROJECT_ROOT / "data" / "cache"))
    parser.add_argument("--max_seq_len", type=int, default=1200)
    args = parser.parse_args()

    build_pocket_graphs(
        dataset_name=args.dataset,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        max_seq_len=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
