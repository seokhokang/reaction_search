import torch
import torch.nn.functional as F
from rdkit import Chem


def euclidean_sim(x, y = None):

    x2 = x.square().sum(dim = 1, keepdim = True)
    if y is None:
        y = x
        y2 = x2
    else:
        y2 = y.square().sum(dim = 1, keepdim = True)
        
    simmat = - (x2 + y2.T - 2 * torch.matmul(x, y.T))
    
    return simmat
 
    
def cosine_sim(x, y = None):

    x = F.normalize(x, p=2, dim=1)
    if y is None:
        y = x
    else:
        y = F.normalize(y, p=2, dim=1)

    simmat = torch.matmul(x, y.T)
    
    return simmat
  
    
def is_valid_smiles(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            return True
        else:
            return False
    except:
        return False