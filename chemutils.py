import os
import numpy as np
from collections import OrderedDict
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS


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


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

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