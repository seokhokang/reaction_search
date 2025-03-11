"""Utility functions.

Functions:
    - euclidean_sim(x, y=None): Computes Euclidean similarity between vectors.
    - cosine_sim(x, y=None): Computes Cosine similarity between vectors.
    - is_valid_smiles(smiles): Validates whether a given SMILES string represents a valid molecule.
    - get_embeddings(trainer, use_dim_reduction, frac_var, batch_size=1024): 
      Extracts molecular embeddings from a trained model, with optional PCA-based dimensionality reduction.
"""

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction

from rdkit import Chem


def euclidean_sim(x, y = None):
    """Computes the Euclidean similarity between vectors.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D).
        y (torch.Tensor, optional): Optional second input tensor of shape (M, D). 
                                    If None, computes self-similarity for x. Default is None.

    Returns:
        torch.Tensor: Similarity matrix of shape (N, M), where higher values indicate greater similarity.
    """
    x2 = x.square().sum(dim = 1, keepdim = True)
    if y is None:
        y = x
        y2 = x2
    else:
        y2 = y.square().sum(dim = 1, keepdim = True)
        
    simmat = - (x2 + y2.T - 2 * torch.matmul(x, y.T))
    
    return simmat
 
    
def cosine_sim(x, y = None):
    """Computes the cosine similarity between vectors.

    Args:
        x (torch.Tensor): Input tensor of shape (N, D).
        y (torch.Tensor, optional): Optional second input tensor of shape (M, D). 
                                    If None, computes self-similarity for x. Default is None.

    Returns:
        torch.Tensor: Similarity matrix of shape (N, M), with values ranging from -1 to 1.
    """
    
    x = F.normalize(x, p=2, dim=1)
    if y is None:
        y = x
    else:
        y = F.normalize(y, p=2, dim=1)

    simmat = torch.matmul(x, y.T)
    
    return simmat
  
    
def is_valid_smiles(smiles):
    """Checks if a given SMILES string is a valid chemical structure.

    Args:
        smiles (str): SMILES representation of a molecule.

    Returns:
        bool: True if the SMILES string is valid, otherwise False.
    """
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            return True
        else:
            return False
    except:
        return False
        

def get_embeddings(trainer, use_dim_reduction, frac_var, batch_size = 1024):
    """Extracts embeddings for chemical reactions using a trained model.

    Args:
        trainer (Trainer): Trained model used to generate embeddings.
        use_dim_reduction (bool): If True, applies dimensionality reduction (PCA).
        frac_var (float): Fraction of variance to retain when performing PCA.
        batch_size (int, optional): Batch size for data loading.

    Returns:
        tuple:
            - np.ndarray: Array of reaction IDs.
            - torch.Tensor: Tensor of reaction embeddings.
            - torch.Tensor or None: principal component matrix (V) if PCA applied, otherwise None.
    """

    # extract embeddings
    ref_rid_list = []
    ref_embed_list = []
    for ref_data in ['uspto_train', 'uspto_valid']:
        ref_loader = DataLoader(
            dataset = ReactionDataset(ref_data, trainer.mol_dict),
            batch_size = batch_size,
            collate_fn = collate_reaction,
            shuffle = False,
            drop_last = False
        )
        
        for batch_idx, batch_data in enumerate(ref_loader):
            rid = np.array(batch_data[0])
            product, product_pred = trainer.embed(batch_data[1], batch_data[2])
            embed = np.concatenate([product, product_pred], axis=1)
    
            ref_rid_list.extend(rid)
            ref_embed_list.append(embed)
  
    ref_rid_list = np.array(ref_rid_list)        
    ref_embed_list = np.vstack(ref_embed_list)
    ref_embed_list = torch.FloatTensor(ref_embed_list)
    V = None

    # dimensionality reduction
    if use_dim_reduction:
        ref_cnt = ref_embed_list.shape[0]
        original_dim = ref_embed_list.shape[1] // 2

        ref_embed_list = torch.cat([ref_embed_list[:,:original_dim], ref_embed_list[:,original_dim:]], 0)
        
        _, S, V = torch.pca_lowrank(ref_embed_list, q=min(original_dim, 64), center=True, niter=2)
        exp_var_ratio = np.cumsum((S**2 / (len(ref_embed_list)-1)) / torch.var(ref_embed_list, dim=0).sum())
        reduced_dim = np.where(exp_var_ratio>frac_var)[0][0] + 1
    
        V = V[:,:reduced_dim]
        
        ref_embed_list = torch.matmul(ref_embed_list, V)
        ref_embed_list = torch.cat([ref_embed_list[:ref_cnt], ref_embed_list[ref_cnt:]], 1)

    return ref_rid_list, ref_embed_list, V