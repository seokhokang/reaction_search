"""Functions to encode chemical properties of atoms and bonds into one-hot vectors and to transform SMILES strings into molecular graphs with node and edge features.
 They leverage RDKit for chemical informatics operations and DGL for graph construction.

Functions:
- one_hot_encoding: Converts a value into a one-hot encoded vector.
- atom_featurizer: Extracts features from an RDKit Atom object.
- bond_featurizer: Extracts features from an RDKit Bond object.
- smi_to_graph: Converts a SMILES string to a DGLGraph with featurized nodes and edges.
"""

import yaml
import numpy as np
import torch
from dgl import graph
from rdkit import Chem, RDConfig, rdBase
rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')

# Load configuration
config_url = 'config.yaml'
with open(config_url, 'r') as f:
    config = yaml.safe_load(f)  

node_in_feats = config['data']['node_in_feats']
edge_in_feats = config['data']['edge_in_feats'] 

# Atom properties
atom_types = list(range(1, 93)) # Atomic numbers 1 to 92

charge_types = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

degree_types = list(range(1, 12))

hybridization_types = [Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3,
                       Chem.rdchem.HybridizationType.SP3D,
                       Chem.rdchem.HybridizationType.SP3D2]

numHs_types = [1, 2, 3, 4, 5]

valence_types = [1, 2, 3, 4, 5, 6, 7]

chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

# Bond properties
bond_types = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

bond_dir_types = [Chem.rdchem.BondDir.ENDUPRIGHT,
                  Chem.rdchem.BondDir.ENDDOWNRIGHT]

bond_stereo_types = [Chem.rdchem.BondStereo.STEREOE,
                     Chem.rdchem.BondStereo.STEREOZ]
                                   
def one_hot_encoding(x, types, verbose = False):
    """Converts a value into a one-hot encoded vector.

    Args:
        x (any): The input value to encode.
        types (list): The list of possible values for encoding.
        verbose (bool, optional): If True, prints missing values. Default is False.

    Returns:
        list: One-hot encoded representation of the input value.
    """
    
    if verbose and x not in types:
        if x !=0 and x != "NONE" and x != "UNSPECIFIED":
            print('missing', x, types)
        
    return list(map(lambda s: x == s, types))


def atom_featurizer(a):
    """Extracts atom-level features from an RDKit Atom object.

    Args:
        a (rdkit.Chem.Atom): RDKit Atom object.

    Returns:
        list: A list of binary features representing atomic properties.
    """

    fea = (one_hot_encoding(a.GetAtomicNum(), atom_types)
           + one_hot_encoding(a.GetFormalCharge(), charge_types)
           + one_hot_encoding(a.GetDegree(), degree_types)
           + one_hot_encoding(a.GetHybridization(), hybridization_types)
           + one_hot_encoding(a.GetTotalNumHs(), numHs_types)
           + one_hot_encoding(a.GetTotalValence(), valence_types)
           + one_hot_encoding(a.GetChiralTag(), chiral_types)
           + [a.GetIsAromatic(), a.IsInRing()]
           + [a.IsInRingSize(s) for s in [3, 4, 5, 6, 7, 8]]
          )

    assert len(fea) == node_in_feats

    return fea


def bond_featurizer(b):
    """Extracts bond-level features from an RDKit Bond object.

    Args:
        b (rdkit.Chem.Bond): RDKit Bond object.

    Returns:
        list: A list of binary features representing bond properties.
    """
    
    fea = (one_hot_encoding(b.GetBondType(), bond_types)
           + one_hot_encoding(b.GetBondDir(), bond_dir_types)
           + one_hot_encoding(b.GetStereo(), bond_stereo_types)
           + [b.IsInRing(), b.GetIsConjugated()]
          )

    assert len(fea) == edge_in_feats

    return fea

   
def smi_to_graph(smi, key = -1):
    """Converts a SMILES string into a molecular graph representation using DGL.

    Args:
        smi (str): SMILES representation of the molecule.
        key (int, optional): Identifier for the molecule (e.g., index). Default is -1 (none).

    Returns:
        dgl.DGLGraph: Graph representation of the molecule, with node and edge features.
    """

    if smi is None or smi == '' or (isinstance(smi, float) and np.isnan(smi)):
        n_node, ndata, edata, src, dst = 0, np.empty((0, node_in_feats), dtype = bool), np.empty((0, edge_in_feats), dtype = bool), [], []
        
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.RemoveHs(mol)
            
            n_node = mol.GetNumAtoms()
            n_edge = mol.GetNumBonds() * 2
            
            ndata = np.array([atom_featurizer(a) for a in mol.GetAtoms()], dtype = bool)
            
            if n_edge > 0:
                edata = np.array([bond_featurizer(b) for b in mol.GetBonds()], dtype = bool)
                edata = np.vstack([edata, edata])
                bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
                src, dst = np.hstack([bond_loc[:,0], bond_loc[:,1]]), np.hstack([bond_loc[:,1], bond_loc[:,0]])
            else:
                edata, src, dst = np.empty((0, edge_in_feats), dtype = bool), [], []
            
        except:
            print('Error processing:', smi)
            n_node, ndata, edata, src, dst = 0, np.empty((0, node_in_feats), dtype = bool), np.empty((0, edge_in_feats), dtype = bool), [], []

    g = graph((src, dst), num_nodes = n_node)
    g.ndata['node_feats'] = torch.FloatTensor(ndata)
    g.edata['edge_feats'] = torch.FloatTensor(edata)
    g.idx = key
    
    return g