import os
import numpy as np
import networkx as nx
import torch
from collections import OrderedDict
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS
from func_group.Substructure_Extraction import get_substructures

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Chem.Kekulize(mol)
    return mol

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    # get the fragment of clique
    Chem.Kekulize(mol)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    Chem.SanitizeMol(mol)
    return new_mol

# --- HimGNN-style functional-group graph features (pure numpy) ---

HIMGNN_ATOM_DIM = 37
HIMGNN_BOND_DIM = 13
HIMGNN_MOTIF_DIM = HIMGNN_ATOM_DIM + HIMGNN_BOND_DIM
HIMGNN_AM_EDGE_DIM = 1
HIMGNN_MM_EDGE_DIM = HIMGNN_ATOM_DIM

_HIMGNN_ATOM_TYPES = ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"]
_HIMGNN_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_HIMGNN_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_HIMGNN_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def _one_hot(value, choices, encode_unknown=False):
    size = len(choices) + int(encode_unknown)
    encoding = [0.0] * size
    if value in choices:
        encoding[choices.index(value)] = 1.0
    elif encode_unknown:
        encoding[-1] = 1.0
    return encoding


def _explicit_valence(atom):
    """Read explicit valence across old and new RDKit Atom APIs."""
    if hasattr(atom, "GetValence"):
        return atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
    return atom.GetExplicitValence()


def himgnn_atom_features(atom):
    """37-dim atom features equivalent to the HimGNN DGL-LifeSci featurizer."""
    cip_code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
    features = (
        _one_hot(atom.GetSymbol(), _HIMGNN_ATOM_TYPES)
        + [float(atom.GetAtomicNum())]
        + _one_hot(_explicit_valence(atom), list(range(7)))
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))
        + _one_hot(atom.GetHybridization(), _HIMGNN_HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic())]
        + [float(atom.IsInRing())]
        + _one_hot(cip_code, ["R", "S"], encode_unknown=True)
        + [float(atom.HasProp("_ChiralityPossible"))]
    )
    assert len(features) == HIMGNN_ATOM_DIM
    return features


def himgnn_bond_features(bond):
    """13-dim bond features equivalent to the HimGNN DGL-LifeSci featurizer."""
    features = (
        _one_hot(bond.GetBondType(), _HIMGNN_BOND_TYPES)
        + [float(bond.IsInRing())]
        + [float(bond.GetIsConjugated())]
        + _one_hot(bond.GetStereo(), _HIMGNN_BOND_STEREOS, encode_unknown=True)
    )
    assert len(features) == HIMGNN_BOND_DIM
    return features


def _functional_group_nx_graph(mol):
    """Build the NetworkX representation consumed by the HimGNN rules."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            atomic_number=torch.tensor([float(atom.GetAtomicNum())]),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=torch.tensor(_one_hot(bond.GetBondType(), _HIMGNN_BOND_TYPES)),
        )
    return graph


def functional_group_decomp(mol, use_cycle=True):
    """
    将分子拆分为 HimGNN 风格的 motif。

    motif 包括识别出的官能团、可选的环，以及没有归入前两类结构的键。
    不同 motif 可以共享原子，例如一个原子可能同时属于环和官能团。
    """
    incidence = get_substructures([_functional_group_nx_graph(mol)], use_cycle=use_cycle)[0]
    motifs = []
    if incidence.numel() > 0:
        atom_indices, motif_indices = incidence.unbind(dim=0)
        for motif_idx in range(int(motif_indices.max().item()) + 1):
            atoms = atom_indices[motif_indices == motif_idx].tolist()
            if atoms:
                motifs.append(sorted(set(atoms)))

    unique_motifs = []
    for motif in motifs:
        if motif not in unique_motifs:
            unique_motifs.append(motif)
    motifs = [
        motif for motif in unique_motifs
        if not any(set(motif) < set(other) for other in unique_motifs)
    ]

    # 此处再补充单原子 motif，确保孤立原子也至少属于一个 motif。
    covered_atoms = {atom for motif in motifs for atom in motif}
    motifs.extend([[atom.GetIdx()] for atom in mol.GetAtoms() if atom.GetIdx() not in covered_atoms])
    return motifs or [list(range(mol.GetNumAtoms()))]


def build_himgnn_mol_hetero_dict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    motifs = functional_group_decomp(mol, use_cycle=True)
    print(f"  Decomposed {smiles} into {len(motifs)} motifs: {motifs}")

    # motif 允许重叠，因此一个原子可能映射到多个 motif。
    atom_to_motifs = {a_idx: [] for a_idx in range(n_atoms)}
    for m_idx, motif_atoms in enumerate(motifs):
        for atom_idx in motif_atoms:
            atom_to_motifs[atom_idx].append(m_idx)

    x_atom = np.array(
        [himgnn_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )

    # 每个 motif 的 50 维特征由两部分拼接：
    # motif 内原子的 37 维特征之和，以及内部化学键的 13 维特征之和。
    x_motif_list = []
    for motif_atoms in motifs:
        motif_set = set(motif_atoms)
        atom_sum = x_atom[motif_atoms].sum(axis=0)
        bond_sum = np.zeros(HIMGNN_BOND_DIM, dtype=np.float32)
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() in motif_set and bond.GetEndAtomIdx() in motif_set:
                bond_sum += np.asarray(himgnn_bond_features(bond), dtype=np.float32)
        x_motif_list.append(np.concatenate([atom_sum, bond_sum]))
    x_motif = np.array(x_motif_list, dtype=np.float32)

    # aa 边：每条无向化学键展开成两个方向，两个方向使用相同的键特征。
    aa_src, aa_dst, aa_attr = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = himgnn_bond_features(bond)
        aa_src.extend([u, v])
        aa_dst.extend([v, u])
        aa_attr.extend([features, features])

    # am 边：记录原子到其所属 motif 的归属关系，边特征固定为 1。
    am_src, am_dst, am_attr = [], [], []
    for atom_idx, motif_indices in atom_to_motifs.items():
        for motif_idx in motif_indices:
            am_src.append(atom_idx)
            am_dst.append(motif_idx)
            am_attr.append([1.0])

    # mm 边分两步建立。若两个 motif 共享原子，边特征是共享原子特征之和；
    # 否则，如果两个 motif 之间存在化学键，则使用该键两端原子特征之和。
    # 字典可以避免同一对 motif 被重叠关系或多条化学键重复加入。
    mm_connections = {}
    motif_sets = [set(motif) for motif in motifs]
    for m_i in range(len(motifs)):
        for m_j in range(m_i + 1, len(motifs)):
            shared_atoms = motif_sets[m_i] & motif_sets[m_j]
            if shared_atoms:
                mm_connections[(m_i, m_j)] = x_atom[sorted(shared_atoms)].sum(axis=0)
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for m_u in atom_to_motifs[u]:
            for m_v in atom_to_motifs[v]:
                if m_u == m_v:
                    continue
                pair = tuple(sorted((m_u, m_v)))
                if pair not in mm_connections:
                    mm_connections[pair] = x_atom[[u, v]].sum(axis=0)

    # 与 aa 边一致，将 motif 间连接展开成两个方向。
    mm_src, mm_dst, mm_attr = [], [], []
    for (m_i, m_j), features in mm_connections.items():
        mm_src.extend([m_i, m_j])
        mm_dst.extend([m_j, m_i])
        mm_attr.extend([features, features])

    print(
        f"  Molecule {smiles}: x_atom={x_atom.shape}, x_motif={x_motif.shape}, "
        f"edges atom-bond-atom={len(aa_src)}, atom-in-motif={len(am_src)}, "
        f"motif-connects-motif={len(mm_src)}"
    )

    return {
        "x_atom": x_atom,
        "x_motif": x_motif,
        "aa_edge_index": np.array([aa_src, aa_dst], dtype=np.int64) if aa_src else np.empty((2, 0), dtype=np.int64),
        "aa_edge_attr": np.array(aa_attr, dtype=np.float32) if aa_attr else np.empty((0, HIMGNN_BOND_DIM), dtype=np.float32),
        "am_edge_index": np.array([am_src, am_dst], dtype=np.int64) if am_src else np.empty((2, 0), dtype=np.int64),
        "am_edge_attr": np.array(am_attr, dtype=np.float32) if am_attr else np.empty((0, HIMGNN_AM_EDGE_DIM), dtype=np.float32),
        "mm_edge_index": np.array([mm_src, mm_dst], dtype=np.int64) if mm_src else np.empty((2, 0), dtype=np.int64),
        "mm_edge_attr": np.array(mm_attr, dtype=np.float32) if mm_attr else np.empty((0, HIMGNN_MM_EDGE_DIM), dtype=np.float32),
    }


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms <= 1:
        return [list(range(n_atoms))]

    # Find BRICS bonds
    res = list(BRICS.FindBRICSBonds(mol))
    if not res:
        return [list(range(n_atoms))]

    # Get the indices of bonds to be broken
    bonds_to_break = [mol.GetBondBetweenAtoms(x[0][0], x[0][1]).GetIdx() for x in res if mol.GetBondBetweenAtoms(x[0][0], x[0][1])]
    if not bonds_to_break:
        return [list(range(n_atoms))]
        
    # Break the molecule at the specified bonds
    fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=False)
    
    # Get the atom indices for each resulting fragment
    cliques = Chem.GetMolFrags(fragmented_mol, asMols=False)

    return [list(c) for c in cliques]
