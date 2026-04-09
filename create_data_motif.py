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
from utils import *
import MDAnalysis as mda
import esm
from protein_process import *
from rdkit.Chem import BRICS, ChemicalFeatures
from enum import Enum
from chemutils import *

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
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}

# --- 2. Motif 和药效团相关函数 ---
class Pharmacophores(Enum):
    Donor = 'Donor'
    Acceptor = 'Acceptor'
    NegIon = 'NegIonizable'
    PosIon = 'PosIonizable'
    ZnB = 'ZnBinder'
    Aromatic = 'Aromatic'
    Hydro = 'Hydrophobe'
    LumHydro = 'LumpedHydrophope'

AllPharmaTypes = [p.value for p in Pharmacophores]
fdef_path = os.path.join(os.path.dirname(__file__), "BaseFeatures.fdef")
feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
pharma_type_to_index = {typ: i for i, typ in enumerate(AllPharmaTypes)}


def smile_to_graph(smile):
    """
    从 SMILES 构建一个包含原子和 Motif 节点的图。
    - 节点特征包含类型标识符 (原子=0, Motif=1)
    - Motif 特征为其8维药效团 multi-hot 编码
    - 包含原子-原子, 原子-Motif两类边
    """
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # print(f"RDKit无法解析SMILES: {smile}")
            return None
    except Exception as e:
        # print(f"SMILES处理异常: {smile}, Error: {e}")
        return None

    # 1. --- 原子节点和原子-原子边 ---
    atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
    x_atoms = np.array(atom_features_list, dtype=np.float32)
    num_atoms = x_atoms.shape[0]

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

    # 2. --- Motif 分解 ---
    cliques = motif_decomp(mol)
    num_motifs = len(cliques)
    
    is_virtual_motif = (num_motifs <= 1)

    if is_virtual_motif:
        # 对于不可分解的分子，我们定义它的唯一基序就是它自身
        cliques = [list(range(num_atoms))]
        num_motifs = 1

    # 3. --- Motif 节点特征计算 (8维 multi-hot) ---
    motif_features_list = []

    # 如果是虚拟 Motif，其特征直接为全零
    if is_virtual_motif:
        feature_vec = np.zeros(len(AllPharmaTypes), dtype=np.float32)
        motif_features_list.append(feature_vec)
        print(f"{smile} create virtual motif")
    # 否则，正常计算每个 Motif 的药效团特征
    else:
        for clique in cliques:
            submol = get_clique_mol(mol, clique)
            feature_vec = np.zeros(len(AllPharmaTypes), dtype=np.float32)
            if submol:
                feats = feature_factory.GetFeaturesForMol(submol)
                for feat in feats:
                    family = feat.GetFamily()
                    if family in pharma_type_to_index:
                        feature_vec[pharma_type_to_index[family]] = 1.0
            motif_features_list.append(feature_vec)

    x_motifs_8d = np.array(motif_features_list, dtype=np.float32)

    # 4. --- 构建统一的节点特征矩阵 X ---
    # 原子特征: [47维特征, 标识符0]
    atom_type_flags = np.zeros((num_atoms, 1), dtype=np.float32)
    x_atoms_final = np.concatenate([x_atoms, atom_type_flags], axis=1)

    # Motif特征: [8维multi-hot, 39维0, 标识符1]
    padding = np.zeros((num_motifs, 47 - 8), dtype=np.float32)
    motif_type_flags = np.ones((num_motifs, 1), dtype=np.float32)
    x_motifs_final = np.concatenate([x_motifs_8d, padding, motif_type_flags], axis=1)

    # 合并所有节点特征
    x = np.concatenate([x_atoms_final, x_motifs_final], axis=0)

    # 5. --- 构建边 ---
    # 原子-Motif 边
    atom_motif_edges = []
    for k, motif in enumerate(cliques):
        motif_node_idx = num_atoms + k
        for atom_node_idx in motif:
            atom_motif_edges.extend([[atom_node_idx, motif_node_idx]])  # 单向边


    # 合并所有边索引
    edge_index = np.concatenate([
        np.array(atom_atom_edges).T if atom_atom_edges else np.empty((2, 0)),
        np.array(atom_motif_edges).T if atom_motif_edges else np.empty((2, 0))
    ], axis=1)

    # 6. --- 构建统一的边特征矩阵 ---
    # 为新增的边类型定义虚拟特征
    # [bond_type=5 (虚拟), is_in_ring=0 (非环)] for atom-motif
    atom_motif_edge_attr = np.array([[5, 0]] * len(atom_motif_edges), dtype=np.float32)

    edge_attr = np.concatenate([
        np.array(atom_atom_edge_features) if atom_atom_edge_features else np.empty((0, 2)),
        atom_motif_edge_attr,
        # motif_motif_edge_attr
    ], axis=0)
    
    num_part = x.shape[0]
    print(f"smiles: {smile}")
    print(f"Total nodes: {x.shape[0]}")
    print(f"Atom Nodes: {num_atoms} Motif Nodes: {num_motifs}")
    print(f"Node features shape: {x.shape}")  # 现在应该是 [total_nodes, 48]
    print(f"Edge features shape: {edge_attr.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'num_part': num_part
    }

# --- New Protein Residue Graph Construction ---
def target_to_graph(protein_pdb, dataset, key, cutoff_distance=10.0, model=None, alphabet=None):
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
        esm_feats = get_esm_embeddings(sequence, model, alphabet)
        
        # Calculate physicochemical features
        res_feats = np.array([calc_res_features(res) for res in protein_residues])
        
        # Load residue surface features
        surface_file = f'data/{dataset}/residue_surface_feat/{key}.pt'
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


# --- Main Data Processing ---
dataset = 'kiba'
seeds = [0, 1, 2, 3, 4]
strategies = ['cold_drug','cold_target','all_cold','random']
# strategies = ['cold_drug']

# 1. Prepare drug features (3D graphs and fingerprints)
compound_iso_smiles = []
target_keys = []
for dt_name in ['kiba']:
    df = pd.read_csv(f'data/{dt_name}/process.csv')
    compound_iso_smiles += list(df['Drug'])
    target_keys += list(df['target_key'])

mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
compound_iso_smiles = set(compound_iso_smiles)
target_keys = set(target_keys)

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
    # print(ecfp_arr.shape)
    fingerprint[smile] = ecfp_arr
    
print("Drug feature processing complete.")

print("Loading ESM-2 model...")
# Load ESM-2 model
model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t12_35M_UR50D")
model.eval()  # disables dropout for deterministic results
print("ESM-2 model loaded.")

protein_graphs = {}
esm_feats = {}

for key in target_keys:
    # --- Process Protein Graph ---
    pocket_dir = f'data/{dataset}/pocket1_{dataset}'
    pocket_file = get_pocket_file(pocket_dir, key)
    if not os.path.exists(pocket_file):
        print(f"PDB file not found: {pocket_file}. Skipping key {key}.")
        continue
    # print("pocket file:", pocket_file)
    graph_data = target_to_graph(pocket_file, dataset, key, model=model, alphabet=alphabet)

    # --- Process Sequence Features ---
    seq_file = f'data/{dataset}/sequence/{key}.npy'
    if not os.path.exists(seq_file):
        print(f"Sequence file not found: {seq_file}. Skipping key {key}.")
        continue

    seq_feat = np.load(seq_file, allow_pickle=True)

    # --- Store in dictionaries ---
    protein_graphs[key] = graph_data
    esm_feats[key] = seq_feat

print("Protein feature processing complete.")

name = "hmol_motif"  
# 2. Process datasets for different splits
for strategy in strategies:
    for seed in seeds:
        split_dir = f'split_data/seed_{seed}/{strategy}'

        if not os.path.exists(f'{split_dir}/{dataset}_train.csv'):
            print(f"Skipping seed {seed}, strategy {strategy} - data files not found")
            continue

        processed_data_dir = f'data/processed/{strategy}/seed_{seed}'
        os.makedirs(processed_data_dir, exist_ok=True)

        processed_data_file_train = f'{processed_data_dir}/{dataset}_train_{name}.pt'
        processed_data_file_val = f'{processed_data_dir}/{dataset}_val_{name}.pt'
        processed_data_file_test = f'{processed_data_dir}/{dataset}_test_{name}.pt'

        if not os.path.isfile(processed_data_file_train) or \
            not os.path.isfile(processed_data_file_val) or \
            not os.path.isfile(processed_data_file_test):

            print(f"Processing dataset: {dataset}, seed: {seed}, strategy: {strategy}")

            # Process training set
            df_train = pd.read_csv(f'{split_dir}/{dataset}_train.csv')
            train_drugs, train_prots, train_Y = list(df_train['Drug']), list(df_train['target_key']), list(df_train['Y'])
            train_target_keys = list(df_train['target_key'])
            print(f"Extracted {len(train_target_keys)} target keys for train set")

            train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
            train_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset}_train_{name}',
                                            xd=train_drugs, xt=train_prots, y=train_Y,
                                            smile_graph=smile_graph,
                                            pocket_graph=protein_graphs, # Now contains residue graphs
                                            fingerprint=fingerprint,
                                            esm_feats=esm_feats) 

            # Process validation set
            df_val = pd.read_csv(f'{split_dir}/{dataset}_val.csv')
            val_drugs, val_prots, val_Y = list(df_val['Drug']), list(df_val['target_key']), list(df_val['Y'])
            val_target_keys = list(df_val['target_key'])
            print(f"Extracted {len(val_target_keys)} target keys for validation set")

            val_drugs, val_prots, val_Y = np.asarray(val_drugs), np.asarray(val_prots), np.asarray(val_Y)
            print(f'Preparing {dataset}_val.pt for seed {seed}, strategy {strategy}!')
            val_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset}_val_{name}',
                                            xd=val_drugs, xt=val_prots, y=val_Y,
                                            smile_graph=smile_graph,
                                            pocket_graph=protein_graphs,
                                            fingerprint=fingerprint,
                                            esm_feats=esm_feats)

            # Process test set
            df_test = pd.read_csv(f'{split_dir}/{dataset}_test.csv')
            test_drugs, test_prots, test_Y = list(df_test['Drug']), list(df_test['target_key']), list(df_test['Y'])
            test_target_keys = list(df_test['target_key'])
            print(f"Extracted {len(test_target_keys)} target keys for test set")
            
            test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)
            print(f'Preparing {dataset}_test.pt for seed {seed}, strategy {strategy}!')
            test_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset}_test_{name}',
                                            xd=test_drugs, xt=test_prots, y=test_Y,
                                            smile_graph=smile_graph,
                                            pocket_graph=protein_graphs,
                                            fingerprint=fingerprint,
                                            esm_feats=esm_feats)

            print(f"Dataset files for seed {seed}, strategy {strategy} have been created")
        else:
            print(f"Dataset files for seed {seed}, strategy {strategy} already exist")
