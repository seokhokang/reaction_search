import numpy as np

import torch
from dgl import graph

from rdkit import Chem, RDConfig, rdBase
rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')


node_in_feats = 140
edge_in_feats = 8 

atom_types = list(range(1, 93))

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

bond_types = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

bond_direction_types = [Chem.rdchem.BondDir.ENDUPRIGHT,
                        Chem.rdchem.BondDir.ENDDOWNRIGHT]
                                   
def one_hot_encoding(x, types, verbose = False):

    if verbose and x not in types:
        if x !=0 and x != "NONE" and x != "UNSPECIFIED":
            print('missing', x, types)
        
    return list(map(lambda s: x == s, types))


def atom_featurizer(a):

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

    return fea


def bond_featurizer(b):

    fea = (one_hot_encoding(b.GetBondType(), bond_types)
           + one_hot_encoding(b.GetBondDir(), bond_direction_types)
           + [b.IsInRing(), b.GetIsConjugated()]
          )
    
    return fea

   
def smi_to_graph(smi, key = -1):
    
    if smi is None or smi == '' or (isinstance(smi, float) and np.isnan(smi)):
        n_node, ndata, edata, src, dst = 0, np.empty((0, node_in_feats), dtype = bool), np.empty((0, edge_in_feats), dtype = bool), [], []
        
    else:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize = False)
            mol = Chem.RemoveHs(mol, sanitize = False)
            
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