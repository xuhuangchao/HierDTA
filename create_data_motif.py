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

# ===== Chemprop Standard Atom Features (134-dim) =====
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
    """chemprop 标准 134维原子特征（覆盖 chemutils 版本）"""
    features = onek_encoding_unk_chemprop(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk_chemprop(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk_chemprop(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk_chemprop(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk_chemprop(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk_chemprop(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]
    return features

# ===== Functional Group Knowledge Embedding =====
fg2emb = pickle.load(open('initial/fg2emb.pkl', 'rb'))

with open('data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    fg_names = [line.split()[0] for line in funcgroups]
    fg_smarts = [Chem.MolFromSmarts(line.split()[1]) for line in funcgroups]
    fg_smarts_valid = [(name, smarts) for name, smarts in zip(fg_names, fg_smarts) if smarts is not None]

def match_fg_for_motif(submol):
    """对 motif 子结构匹配官能团，返回匹配的 fg2emb 列表"""
    if submol is None:
        return []
    matched_embs = []
    for name, smarts in fg_smarts_valid:
        if submol.HasSubstructMatch(smarts):
            matched_embs.append(fg2emb[name])
    return matched_embs

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),  #元素周期表序号
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],    #原子的手性
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    # 'possible_bond_dirs': [  # only for double bond stereo information
    #     Chem.rdchem.BondDir.NONE,
    #     Chem.rdchem.BondDir.ENDUPRIGHT,
    #     Chem.rdchem.BondDir.ENDDOWNRIGHT
    # ],
    'possible_bond_inring': [None, False, True]
}

def smile_to_graph(smile):
    """
    从 SMILES 构建一个包含原子和 Motif 节点的图。
    - 节点特征包含类型标识符 (原子=0, Motif=1)
    - 原子特征为 chemprop 标准 134维特征
    - Motif 特征为 133维 fg2emb 官能团嵌入聚合
    - 包含原子-原子, 原子-Motif两类边
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
    except Exception as e:
        return None

    # 1. --- 原子节点和原子-原子边 ---
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x_atoms = np.array(atom_features_list, dtype=np.float32)  # [N_atoms, 134]
    num_atoms = x_atoms.shape[0]
    
    print(f"x_atoms.shape: {x_atoms.shape}")

    atom_atom_edges = []
    atom_atom_edge_features = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_feature = [
            allowable_features['possible_bonds'].index(bond.GetBondType()),
            allowable_features['possible_bond_inring'].index(bond.IsInRing())
        ]
        atom_atom_edges.extend([[u, v], [v, u]])
        atom_atom_edge_features.extend([edge_feature, edge_feature])

    # 2. --- Motif 分解 (BRICS) ---
    cliques = motif_decomp(mol)
    num_motifs = len(cliques)

    is_virtual_motif = (num_motifs <= 1)
    if is_virtual_motif:
        cliques = [list(range(num_atoms))]
        num_motifs = 1

    # 3. --- Motif 节点特征计算 (133维 fg2emb 聚合) ---
    motif_features_list = []
    for clique in cliques:
        submol = get_clique_mol(mol, clique)
        matched_embs = match_fg_for_motif(submol)
        if len(matched_embs) > 0:
            feature_vec = np.mean(matched_embs, axis=0).astype(np.float32)
        else:
            feature_vec = np.zeros(133, dtype=np.float32)
        motif_features_list.append(feature_vec)

    x_motifs = np.array(motif_features_list, dtype=np.float32)  # [N_motifs, 133]
    print(f"x_motifs.shape: {x_motifs.shape}")

    # 4. --- 构建统一的节点特征矩阵 X ---
    # 统一维度: 133 + 1(类型标识符) = 134
    atom_type_flags = np.zeros((num_atoms, 1), dtype=np.float32)
    x_atoms_final = np.concatenate([x_atoms, atom_type_flags], axis=1)  # [N_atoms, 134]

    motif_type_flags = np.ones((num_motifs, 1), dtype=np.float32)
    x_motifs_final = np.concatenate([x_motifs, motif_type_flags], axis=1)  # [N_motifs, 134]

    x = np.concatenate([x_atoms_final, x_motifs_final], axis=0)  # [N_total, 134]

    # 5. --- 构建边 (保持不变) ---
    atom_motif_edges = []
    for k, motif in enumerate(cliques):
        motif_node_idx = num_atoms + k
        for atom_node_idx in motif:
            atom_motif_edges.extend([[atom_node_idx, motif_node_idx],[motif_node_idx, atom_node_idx]])

    edge_index = np.concatenate([
        np.array(atom_atom_edges).T if atom_atom_edges else np.empty((2, 0)),
        np.array(atom_motif_edges).T if atom_motif_edges else np.empty((2, 0))
    ], axis=1)

    atom_motif_edge_attr = np.array([[5, 0]] * len(atom_motif_edges), dtype=np.float32)
    edge_attr = np.concatenate([
        np.array(atom_atom_edge_features) if atom_atom_edge_features else np.empty((0, 2)),
        atom_motif_edge_attr,
    ], axis=0)

    num_part = x.shape[0]

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_part': num_part
    }

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

    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        if g is not None:
            smile_graph[smile] = g

        mol = Chem.MolFromSmiles(smile)
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
