import os
import numpy as np
import pandas as pd
import json, pickle
from collections import OrderedDict
from rdkit import Chem, DataStructs
from rdkit.Chem import MolFromSmiles, AllChem
import networkx as nx
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import torch
import torch.nn as nn
import argparse
from utils import *
import MDAnalysis as mda
import esm
from protein_process import *
from rdkit.Chem import BRICS, ChemicalFeatures
from enum import Enum
from chemutils import *

# ===== ElementKG 本体原子嵌入 (133-dim) =====
ele2emb = pickle.load(open('initial/ele2emb.pkl', 'rb'))

# ===== Chemprop Standard Atom Features =====
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk_chemprop(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def atom_features(atom):
    """
    chemprop 133维 one-hot + ElementKG ele2emb 133维 = 266维原子特征
    """
    chem = onek_encoding_unk_chemprop(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk_chemprop(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk_chemprop(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk_chemprop(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk_chemprop(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk_chemprop(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]

    emb = ele2emb.get(atom.GetAtomicNum() - 1, np.zeros(133, dtype=np.float32)).tolist()

    return chem + emb  # 133 + 133 = 266

# ===== Chemprop Standard Bond Features (14-dim) =====
BOND_FDIM = 14

def bond_features(bond):
    """chemprop 标准 14维键特征"""
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk_chemprop(int(bond.GetStereo()), list(range(6)))
    return fbond


def smile_to_graph(smile, verbose=False):
    """
    从 SMILES 构建纯原子拓扑图。
    - 原子特征: chemprop one-hot (133-d) + ele2emb (133-d) = 266-d
    - 边特征: chemprop 14维键特征
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None, None
    except Exception as e:
        return None, None

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # 1. 原子节点
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features_list, dtype=np.float32)  # [N_atoms, 266]

    # 2. 原子-原子边 (化学键)
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = bond_features(bond)
        edges.extend([[u, v], [v, u]])
        edge_features.extend([ef, ef])

    edge_index = np.array(edges).T if edges else np.empty((2, 0))
    edge_attr = np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, BOND_FDIM))

    if verbose:
        # --- 特征数值检查 ---
        print(f"\n{'='*60}")
        print(f"SMILES: {smile}")
        print(f"  原子数: {n_atoms}, 化学键数: {n_bonds}")
        print(f"  x 形状: {x.shape}, dtype: {x.dtype}")
        print(f"  x[0] 前 10 维 (chemprop one-hot):  {x[0, :10]}")
        print(f"  x[0] 后 5 维 (ele2emb 尾部):      {x[0, -5:]}")
        print(f"  x 非零比例: {np.count_nonzero(x)/x.size:.3f}")
        print(f"  edge_index 形状: {edge_index.shape}")
        print(f"  edge_attr 形状: {edge_attr.shape}")
        if edge_attr.shape[0] > 0:
            print(f"  edge_attr[0]: {edge_attr[0]}")
        print(f"{'='*60}\n")

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_part': x.shape[0],
    }, mol


# --- New Protein Residue Graph Construction ---
def target_to_graph(protein_pdb, dataset, key, cutoff_distance=10.0, model=None, alphabet=None, surface_k=5, device=None):
    try:
        u = mda.Universe(protein_pdb)

        protein_residues = u.residues
        num_residues = len(protein_residues)
        if num_residues == 0:
            print(f"Warning: No residues found in {protein_pdb}")
            return None

        residue_coords = []
        for res in protein_residues:
            # Try to find the Alpha-carbon (CA)
            ca_atoms = res.atoms.select_atoms("name CA")
            if len(ca_atoms) > 0:
                # If CA is found, use its coordinate
                coord = ca_atoms.positions[0]
            else:
                # If no CA is found (e.g., for ligands or other non-amino acid residues),
                # calculate the geometric center of all atoms in the residue.
                coord = res.atoms.center_of_geometry()
            residue_coords.append(coord)
        residue_coords = np.array(residue_coords)

        sequence = ''.join([get_aa_code(res.resname) for res in protein_residues])

        # Get ESM embeddings
        esm_feats = get_esm_embeddings(sequence, model, alphabet, device=device)

        # Calculate physicochemical features
        res_feats = np.array([calc_res_features(res) for res in protein_residues])

        # Load residue surface features
        surface_file = f'data/{dataset}/preprocessed/residue_surface/k{surface_k}/{key}.pt'
        data = torch.load(surface_file, map_location='cpu')
        target_surface = data["residue_feat_from_surface"]
        if target_surface.shape[0] != num_residues:
            print(f"Warning: Mismatch in residue count for {key}. PDB: {num_residues}, Surface: {target_surface.shape[0]}. Skipping.")
            return None

        # Combine node features, including surface features
        node_features = np.concatenate((res_feats, esm_feats, target_surface), axis=1)

        # Build graph edges
        edgeids, distm = obtain_edge(u, cutoff_distance)
        if len(edgeids) == 0:
            print(f"Warning: No edges found for {protein_pdb}. Check cutoff or structure.")
            return num_residues, node_features, np.array([[], []]), residue_coords

        src_list, dst_list = zip(*edgeids)
        edge_index = np.array([src_list, dst_list])

        return num_residues, node_features, edge_index, residue_coords

    except Exception as e:
        print(f"Error processing protein {protein_pdb} with key {key}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Cache Building (与 seed/strategy 无关，每个 dataset 只需运行一次) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build global feature caches for DTA datasets')
    parser.add_argument('--dataset', type=str, default='kiba', choices=['kiba', 'davis'],
                        help='Dataset name to process')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing cache files')
    parser.add_argument('--surface_k', type=int, default=5,
                        help='k for surface-to-residue mapping (must match surface_process.py)')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='GPU index to use for ESM inference')
    args = parser.parse_args()

    dataset = args.dataset
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        'smile_graph': f'{cache_dir}/{dataset}_smile_graph.pt',
        'fingerprint': f'{cache_dir}/{dataset}_fingerprint.pt',
        'protein_graphs': f'{cache_dir}/{dataset}_protein_graphs_k{args.surface_k}.pt',
        'esm_feats': f'{cache_dir}/{dataset}_esm_feats.pt',
    }

    # 检查 cache 是否已存在
    if not args.force and all(os.path.isfile(f) for f in cache_files.values()):
        print(f"All cache files for {dataset} already exist. Use --force to overwrite.")
        exit(0)

    # 1. Prepare drug features (graphs and fingerprints)
    print(f"Building drug feature cache for {dataset}...")
    df = pd.read_csv(f'data/{dataset}/process.csv')
    compound_iso_smiles = set(df['Drug'].tolist())

    mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    smile_graph = {}
    fingerprint = {}
    verbose_count = 0

    for smile in compound_iso_smiles:
        verbose = (verbose_count < 3)
        g, mol = smile_to_graph(smile, verbose=verbose)
        if verbose:
            verbose_count += 1
        if g is not None:
            smile_graph[smile] = g

        if mol is None:
            continue
        ecfp_arr = mg.GetFingerprintAsNumPy(mol)
        fingerprint[smile] = ecfp_arr

    torch.save(smile_graph, cache_files['smile_graph'])
    torch.save(fingerprint, cache_files['fingerprint'])
    print(f"Drug cache saved: {cache_files['smile_graph']}")
    print(f"  Unique drugs: {len(smile_graph)}")

    # 2. Prepare protein features
    print(f"Building protein feature cache for {dataset}...")
    target_keys = set(df['target_key'].tolist())

    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"ESM-2 model loaded on {device}.")

    protein_graphs = {}
    esm_feats = {}

    for key in target_keys:
        pocket_dir = f'data/{dataset}/pocket1_{dataset}'
        pocket_file = get_pocket_file(pocket_dir, key)
        if not os.path.exists(pocket_file):
            print(f"PDB file not found: {pocket_file}. Skipping key {key}.")
            continue
        graph_data = target_to_graph(pocket_file, dataset, key, model=model, alphabet=alphabet, surface_k=args.surface_k, device=device)

        seq_file = f'data/{dataset}/preprocessed/sequence/{key}.npy'
        if not os.path.exists(seq_file):
            print(f"Sequence file not found: {seq_file}. Skipping key {key}.")
            continue

        seq_feat = np.load(seq_file, allow_pickle=True)

        protein_graphs[key] = graph_data
        esm_feats[key] = seq_feat

    torch.save(protein_graphs, cache_files['protein_graphs'])
    torch.save(esm_feats, cache_files['esm_feats'])
    print(f"Protein cache saved: {cache_files['protein_graphs']}")
    print(f"  Unique proteins: {len(protein_graphs)}")
    print(f"\nAll caches for {dataset} built successfully.")
    print(f"Cache directory: {cache_dir}")
