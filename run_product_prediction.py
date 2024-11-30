import os, sys
import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction, collate_product

from model import GINPredictor
from train import Trainer

from util import euclidean_sim

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--batch_size', '-b', type=int, default=4096)
parser.add_argument('--dim_reduction', '-r', type=int, default=1)

args = parser.parse_args()


# configurations
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
batch_size = args.batch_size
use_dim_reduction = (args.dim_reduction == 1)

with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

graph_net = GINPredictor(node_in_feats=140, edge_in_feats=8)


# load the representation model
model_path = './model/%s_checkpoint.pt'%identifier

trainer = Trainer(graph_net, model_path, mol_dict, cuda)
trainer.load(model_path)


# extract product embeddings
product_path = './embed/%s_test_product_embeddings.npz'%identifier

if os.path.exists(product_path):
    data = np.load(product_path)
    product_pid_list = data['ids']
    product_embed_list = data['embeds']

else:
    product_data = ReactionDataset('uspto_test', mol_dict, mode = 'product')
    
    product_loader = DataLoader(
        dataset = product_data,
        batch_size = batch_size,
        collate_fn = collate_product,
        shuffle = False,
        drop_last = False
    )
     
    product_pid_list = []
    product_embed_list = []
    for batch_idx, batch_data in enumerate(product_loader):
    
        pid = np.array(batch_data[0])
        product, _ = trainer.embed(batch_data[1])
    
        product_pid_list.extend(pid)
        product_embed_list.append(product)
    
    product_pid_list = np.array(product_pid_list)        
    product_embed_list = np.vstack(product_embed_list)
        
    np.savez_compressed(product_path, ids = product_pid_list, embeds = product_embed_list)


# extract query embeddings
query_path = './embed/%s_test_query_embeddings.npz'%identifier
query_data = ReactionDataset('uspto_test', mol_dict)
query_dict = query_data.data

if os.path.exists(query_path):
    data = np.load(query_path)
    query_rid_list = data['ids']
    query_embed_list = data['embeds']

else:
    query_loader = DataLoader(
        dataset = query_data,
        batch_size = batch_size,
        collate_fn = collate_reaction,
        shuffle = False,
        drop_last = False
    )
    
    query_rid_list = []
    query_embed_list = []
    for batch_idx, batch_data in enumerate(query_loader):
    
        rid = np.array(batch_data[0])
        _, product_pred = trainer.embed(None, batch_data[2])
    
        query_rid_list.extend(rid)
        query_embed_list.append(product_pred)
    
    query_rid_list = np.array(query_rid_list)        
    query_embed_list = np.vstack(query_embed_list)
    
    np.savez_compressed(query_path, ids = query_rid_list, embeds = query_embed_list)


# prepare   
query_embed_list = torch.FloatTensor(query_embed_list).to(cuda)
product_embed_list = torch.FloatTensor(product_embed_list).to(cuda)
if use_dim_reduction:
    print('use dim reduction')
    pca = np.load('./embed/%s_pca.npz'%identifier)
    V = torch.FloatTensor(pca['pc']).to(cuda)
    
    query_embed_list = torch.matmul(query_embed_list, V)
    product_embed_list = torch.matmul(product_embed_list, V)


# product prediction performance evaluation
rank_list = []
for i, rid in enumerate(query_rid_list):

    answer_pid = query_dict[rid]['product'][0]

    sim = euclidean_sim(query_embed_list[i:i+1], product_embed_list).cpu().numpy().ravel()
    sort_idx = np.argsort(-sim)

    rank = np.where(product_pid_list[sort_idx] == answer_pid)[0][0] + 1
    rank_list.append(rank)
    
rank_list = np.array(rank_list)

mrr = np.mean(1/rank_list)
mr = np.mean(rank_list)
print('SUMMARY MRR %.3f MR %.3f'%(mrr,mr))

for K in [1, 3, 5, 10]: 
    print('SUMMARY FRAC TOP-%d RANKED: %.3f'%(K, np.mean(rank_list<=K)))