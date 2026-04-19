"""
features.py - Molecule to graph conversion for GNN model.
Converts SMILES strings to PyTorch Geometric Data objects
with atom and bond features including Gasteiger charges.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# Atom feature vocabularies
ATOMIC_NUMS     = list(range(1, 119))
DEGREES         = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATIONS  = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
FORMAL_CHARGES  = [-2, -1, 0, 1, 2]
NUM_HS          = [0, 1, 2, 3, 4]


def one_hot(value, choices):
    """One-hot encode value against choices list, with unknown bucket."""
    encoding = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    encoding[idx] = 1
    return encoding


def get_atom_features(atom):
    """
    Extract feature vector for a single atom.
    Includes topology, hybridization, charge, and Gasteiger partial charge.
    
    Returns list of floats, length = ATOM_FEATURE_DIM
    """
    # Gasteiger charge already computed on owning mol
    try:
        gasteiger = atom.GetDoubleProp('_GasteigerCharge')
        if not np.isfinite(gasteiger):
            gasteiger = 0.0
    except:
        gasteiger = 0.0

    return (
        one_hot(atom.GetAtomicNum(),    ATOMIC_NUMS)    +  # 119
        one_hot(atom.GetDegree(),       DEGREES)         +  # 8
        one_hot(atom.GetHybridization(), HYBRIDIZATIONS) +  # 7
        one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)  +  # 6
        one_hot(atom.GetTotalNumHs(),   NUM_HS)          +  # 6
        [
            int(atom.GetIsAromatic()),   # 1
            int(atom.IsInRing()),        # 1
            gasteiger,                   # 1 - continuous charge
            atom.GetMass() / 100.0,      # 1 - normalized mass
        ]
    )
    # Total: 119 + 8 + 7 + 6 + 6 + 4 = 150 features


ATOM_FEATURE_DIM = 150

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def get_bond_features(bond):
    """
    Extract feature vector for a single bond.
    
    Returns list of floats, length = BOND_FEATURE_DIM
    """
    return (
        one_hot(bond.GetBondType(), BOND_TYPES) +  # 5
        [
            int(bond.GetIsConjugated()),            # 1
            int(bond.IsInRing()),                   # 1
            int(bond.GetStereo() !=
                Chem.rdchem.BondStereo.STEREONONE), # 1
        ]
    )
    # Total: 5 + 3 = 8 features


BOND_FEATURE_DIM = 8


def mol_to_graph(smiles, y=None):
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        y: target value (pIC50), optional
    
    Returns:
        torch_geometric.data.Data object, or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Compute Gasteiger charges before extracting atom features
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        pass

    # Node features
    atom_features = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index and edge features (bidirectional)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:
        edge_index = [[0, 0]]
        edge_attr  = [[0] * BOND_FEATURE_DIM]

    data = Data(
        x          = x,
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float),
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


def smiles_to_graphs(df):
    """
    Convert a DataFrame of SMILES to a list of graph Data objects.
    """
    graphs = []
    failed = 0

    for _, row in df.iterrows():
        g = mol_to_graph(row['canonical_smiles'], row['pchembl_value'])
        if g is not None:
            graphs.append(g)
        else:
            failed += 1

    print(f"Converted {len(graphs)} molecules to graphs ({failed} failed)")
    return graphs, failed
