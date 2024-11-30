import os, sys
import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction, query_to_vec

from model import GINPredictor
from train import Trainer

from util import euclidean_sim, is_valid_smiles

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--batch_size', '-b', type=int, default=4096)
parser.add_argument('--frac_var', '-f', type=float, default=0.95)
parser.add_argument('--dim_reduction', '-r', type=int, default=1)

args = parser.parse_args()


# query for search
query = {
    'Q.ID': 1,
    'product': 'c1ccc(-c2ccc3nnc(CNc4ncnc5nc[nH]c45)n3n2)cc1',#smiles
    'reactant': 'Clc1ncnc2nc[nH]c12.NCc1nnc2ccc(-c3ccccc3)nn12',#smiles
}


# configurations
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
batch_size = args.batch_size
frac_var = args.frac_var
use_dim_reduction = (args.dim_reduction == 1)

with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

graph_net = GINPredictor(node_in_feats=140, edge_in_feats=8)


def get_embeddings():

    # extract embeddings
    ref_rid_list = []
    ref_embed_list = []
    for loader in [ref_train_loader, ref_valid_loader]:
        for batch_idx, batch_data in enumerate(loader):
            rid = np.array(batch_data[0])
            product, product_pred = trainer.embed(batch_data[1], batch_data[2])
            embed = np.concatenate([product, product_pred], axis=1)
    
            ref_rid_list.extend(rid)
            ref_embed_list.append(embed)
  
    ref_rid_list = np.array(ref_rid_list)        
    ref_embed_list = np.vstack(ref_embed_list)

    
    # dimensionality reduction
    if use_dim_reduction:
        ref_cnt = ref_embed_list.shape[0]
        original_dim = ref_embed_list.shape[1] // 2
        
        ref_embed_list = torch.FloatTensor(ref_embed_list)
        
        ref_embed_list = torch.cat([ref_embed_list[:,:original_dim], ref_embed_list[:,original_dim:]], 0)
        
        _, S, V = torch.pca_lowrank(ref_embed_list, q=64, center=True, niter=2)
        exp_var_ratio = np.cumsum((S**2 / (len(ref_embed_list)-1)) / torch.var(ref_embed_list, dim=0).sum())
        reduced_dim = np.where(exp_var_ratio>frac_var)[0][0] + 1
    
        V = V[:,:reduced_dim]
        
        ref_embed_list = torch.matmul(ref_embed_list, V)
        ref_embed_list = torch.cat([ref_embed_list[:ref_cnt], ref_embed_list[ref_cnt:]], 1)
        V = V.cpu()
    
        
    ref_embed_list = ref_embed_list.cpu()

    return ref_rid_list, ref_embed_list, V
    
    
# load the representation model
model_path = './model/%s_checkpoint.pt'%identifier

trainer = Trainer(graph_net, model_path, mol_dict, cuda)
trainer.load(model_path)


# extract reaction embeddings
if os.path.exists('./embed/%s_embeddings.npz'%identifier) and os.path.exists('./embed/%s_embeddings_reduced.npz'%identifier):

    if use_dim_reduction:
        pca = np.load('./embed/%s_pca.npz'%identifier)
        V = pca['pc']
        
        data = np.load('./embed/%s_embeddings_reduced.npz'%identifier)
        ref_rid_list = data['ids']
        ref_embed_list = data['embeds'] 
        
        V = torch.FloatTensor(V)
    
    else:
        data = np.load('./embed/%s_embeddings.npz'%identifier)
        ref_rid_list = data['ids']
        ref_embed_list = data['embeds']

    ref_embed_list = torch.FloatTensor(ref_embed_list)

else:

    ref_train_loader = DataLoader(
        dataset = ReactionDataset('uspto_train', mol_dict),
        batch_size = batch_size,
        collate_fn = collate_reaction,
        shuffle = False,
        drop_last = False
    )
    
    ref_valid_loader = DataLoader(
        dataset = ReactionDataset('uspto_valid', mol_dict),
        batch_size = batch_size,
        collate_fn = collate_reaction,
        shuffle = False,
        drop_last = False
    )

    ref_rid_list, ref_embed_list, V = get_embeddings()


ref_data = ReactionDataset('uspto_train', mol_dict, mode = 'dict').data | ReactionDataset('uspto_valid', mol_dict, mode = 'dict').data
assert len(ref_data) == len(ref_embed_list)


# check query
has_product = is_valid_smiles(query['product'])
has_reactant = is_valid_smiles(query['reactant'])
if len(query['product']) > 0: assert has_product
if len(query['reactant']) > 0: assert has_reactant
assert has_product or has_reactant


# retrieve relevant records
q_vec = query_to_vec(query)

q_product, q_product_pred = trainer.embed(q_vec[1], q_vec[2], to_numpy = False)
if not has_reactant: q_product_pred = q_product
if not has_product: q_product = q_product_pred
assert torch.sum(torch.abs(q_product)) > 0 and torch.sum(torch.abs(q_product_pred)) > 0 

if use_dim_reduction:
    q_product = torch.matmul(q_product, V)
    q_product_pred = torch.matmul(q_product_pred, V)
    
q_embed = torch.cat([q_product, q_product_pred], dim=1)
sim = euclidean_sim(q_embed, ref_embed_list).numpy().ravel()
sort_idx = np.argsort(-sim)

print('----------') 
print(query)
print('----------')

retrieved_dict = {}
for i, idx in enumerate(sort_idx):
    rid = ref_rid_list[idx]
    
    inst = ref_data[rid]

    res_product = mol_dict.get(inst['product'][0])
    res_reactant = '.'.join([mol_dict.get(x) for x in inst['reactant']])
    key = '%s_%s'%(res_product, res_reactant)
    if key in retrieved_dict.keys():
        continue
    else:
        retrieved_dict[key] = [query['Q.ID'], query['product'], query['reactant'], res_product, res_reactant]

    print('RANK %d; RX.ID %s; {product: %s, reactant: %s}; dist %.3f'%(len(retrieved_dict), rid, res_product, res_reactant, np.sqrt(-sim[idx])))

    if len(retrieved_dict) == 30: break