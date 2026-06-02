"""Protein pocket residue feature helpers used during preprocessing."""

import glob
import os
import re
from itertools import permutations

import numpy as np
from MDAnalysis.analysis import distances


METAL = [
    "LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI",
    "PT", "ZN", "CO", "PD", "CR", "FE", "V", "MN", "HG", "GA", "CD",
    "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN",
    "Y", "LA", "CE", "PR", "ND", "GD", "TB", "DY", "ER", "TM", "LU",
    "HF", "ZR", "U", "PU", "TH",
]

RESIDUE_TYPES = [
    "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR",
    "TRP", "SER", "THR", "CYS", "MET", "ASN", "GLN", "ASP",
    "GLU", "LYS", "ARG", "HIS", "MSE", "CSO", "PTR", "TPO",
    "KCX", "CSD", "SEP", "MLY", "PCA", "LLP", "M", "X",
]

AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encode x, mapping unknown values to the last bucket."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_resname(res):
    resname = res.resname.strip()
    if res.resname[:2] in {"CA", "FE", "CU"}:
        resname = res.resname[:2]
    return "M" if resname in METAL else resname


def obtain_self_dist(res):
    try:
        atoms = res.atoms
        dists = distances.self_distance_array(atoms.positions)
        ca = atoms.select_atoms("name CA")
        c = atoms.select_atoms("name C")
        n = atoms.select_atoms("name N")
        o = atoms.select_atoms("name O")
        return [
            dists.max() * 0.1,
            dists.min() * 0.1,
            distances.dist(ca, o)[-1][0] * 0.1,
            distances.dist(o, n)[-1][0] * 0.1,
            distances.dist(n, c)[-1][0] * 0.1,
        ]
    except Exception:
        return [0, 0, 0, 0, 0]


def obtain_dihedral_angles(res):
    try:
        phi = res.phi_selection().dihedral.value() if res.phi_selection() is not None else 0
        psi = res.psi_selection().dihedral.value() if res.psi_selection() is not None else 0
        omega = res.omega_selection().dihedral.value() if res.omega_selection() is not None else 0
        chi1 = res.chi1_selection().dihedral.value() if res.chi1_selection() is not None else 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except Exception:
        return [0, 0, 0, 0]


def calc_res_features(res):
    """Return 41 residue features: 32 type + 5 internal distances + 4 angles."""
    return np.array(
        one_of_k_encoding_unk(obtain_resname(res), RESIDUE_TYPES)
        + obtain_self_dist(res)
        + obtain_dihedral_angles(res),
        dtype=np.float32,
    )


def calc_dist(res1, res2):
    return distances.distance_array(res1.atoms.positions, res2.atoms.positions)


def obtain_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax], dtype=np.float32).T


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)

def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0

    # DogSite3 pocket PDB files usually omit CONECT records, so MDAnalysis
    # cannot provide bond topology. Infer peptide connectivity conservatively
    # from chain identity, consecutive residue IDs, and the backbone C-N bond.
    left, right = sorted((i, j))
    res_left = u.residues[left]
    res_right = u.residues[right]

    left_chain_ids = getattr(res_left.atoms, "chainIDs", [])
    right_chain_ids = getattr(res_right.atoms, "chainIDs", [])
    left_chain = left_chain_ids[0] if len(left_chain_ids) > 0 else res_left.segid
    right_chain = right_chain_ids[0] if len(right_chain_ids) > 0 else res_right.segid
    if left_chain != right_chain or res_right.resid - res_left.resid != 1:
        return 0

    left_c = res_left.atoms.select_atoms("name C")
    right_n = res_right.atoms.select_atoms("name N")
    if len(left_c) == 0 or len(right_n) == 0:
        return 0

    peptide_distance = np.linalg.norm(left_c.positions[0] - right_n.positions[0])
    return int(peptide_distance <= 1.8)
        
def get_aa_code(three_letter_code):
    return AA3_TO_AA1.get(three_letter_code, "X")


def get_pocket_file(pocket_dir, key, suffix=".pdb"):
    if not os.path.exists(pocket_dir):
        print(f"Warning: pocket directory does not exist: {pocket_dir}")
        return None

    processed_key = re.sub(r"[.\-() ]", "", key.lower())
    all_pdb_files = glob.glob(os.path.join(pocket_dir, f"{processed_key}*{suffix}"))
    return all_pdb_files[0] if all_pdb_files else None
