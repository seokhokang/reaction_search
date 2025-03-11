"""Train a reaction representation model and extract embeddings.

This script:
- Loads a reaction dataset and trains a GIN-based representation model.
- Extracts embeddings and optionally performs dimensionality reduction.

Usage example (identifier: 'paper', dim_reduction: True, embed_dim: 512):
    python run_train.py -t paper -r 1 -d 512
"""

import os, sys, yaml
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction
from model import GINPredictor
from train import Trainer
from argparse import ArgumentParser

# ------------------------------
# Configurations and Setup
# ------------------------------
config_url = 'config.yaml'
with open(config_url, 'r') as f:
    config = yaml.safe_load(f)  

parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--dim_reduction', '-r', type=int, default=1)
parser.add_argument('--embed_dim', '-d', type=int, default=512)
args = parser.parse_args()

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
use_dim_reduction = (args.dim_reduction == 1)
embed_dim = args.embed_dim

node_in_feats = config['data']['node_in_feats']
edge_in_feats = config['data']['edge_in_feats']
temp_tau = config['train']['tau']
batch_size = config['train']['batch_size']
frac_var = config['pca']['frac_var']

# Ensure directories exist
if not os.path.exists('./model/'): os.makedirs('./model/')
if not os.path.exists('./embed/'): os.makedirs('./embed/')

# Load molecular dictionary
with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

# ------------------------------
# Train the Representation Model
# ------------------------------
model_path = './model/%s_checkpoint.pt'%identifier
graph_net = GINPredictor(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats, project_dim=embed_dim)
trainer = Trainer(graph_net, model_path, mol_dict, cuda, tau = temp_tau)

print('Training model...')

train_loader = DataLoader(
    dataset = ReactionDataset('uspto_train', mol_dict),
    batch_size = batch_size,
    collate_fn = collate_reaction,
    shuffle = True,
    drop_last = True
)

valid_loader = DataLoader(
    dataset = ReactionDataset('uspto_valid', mol_dict),
    batch_size = batch_size,
    collate_fn = collate_reaction,
    shuffle = False,
    drop_last = True
)

trainer.train(train_loader, valid_loader)

# ------------------------------
# Extract and Save Embeddings
# ------------------------------
embed_path = './embed/%s_embeddings.npz'%identifier

print('Extracting embeddings...')

ref_rid_list = []
ref_embed_list = []
for ref_data in ['uspto_train', 'uspto_valid']:
    ref_loader = DataLoader(
        dataset = ReactionDataset(ref_data, mol_dict),
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
ref_embed_list = np.vstack(ref_embed_list).astype(np.float16)
    
print('Embeddings shape:', ref_rid_list.shape, ref_embed_list.shape)
np.savez_compressed(embed_path, ids = ref_rid_list, embeds = ref_embed_list)
        
# ------------------------------
# Dimensionality Reduction (PCA)
# ------------------------------
if use_dim_reduction:

    reduced_path = './embed/%s_embeddings_reduced.npz'%identifier
    pca_path = './embed/%s_pca.npz'%identifier
    
    print('Performing dimensionality reduction...')
    
    ref_cnt = ref_embed_list.shape[0]
    original_dim = ref_embed_list.shape[1] // 2
    
    ref_embed_list = torch.FloatTensor(ref_embed_list)
    ref_embed_list = torch.cat([ref_embed_list[:,:original_dim], ref_embed_list[:,original_dim:]], 0)
    
    _, S, V = torch.pca_lowrank(ref_embed_list, q=min(original_dim, 128), center=True, niter=2)
    exp_var_ratio = np.cumsum((S**2 / (len(ref_embed_list)-1)) / torch.var(ref_embed_list, dim=0).sum())
    reduced_dim = np.where(exp_var_ratio>frac_var)[0][0] + 1
    
    print('Explained variance ratio:', exp_var_ratio)
    print('Reduced dimension:', reduced_dim)
    print('Retained variance:', exp_var_ratio[:reduced_dim])
    
    V = V[:,:reduced_dim]
    
    ref_embed_list = torch.matmul(ref_embed_list, V)
    ref_embed_list = torch.cat([ref_embed_list[:ref_cnt], ref_embed_list[ref_cnt:]], 1)
    ref_embed_list = ref_embed_list.cpu().numpy().astype(np.float16)
    
    V = V.cpu().numpy()
    
    print('PC matrix shape:', V.shape)
    np.savez_compressed(pca_path, pc = V)
    
    print('Reduced embeddings shape:', ref_rid_list.shape, ref_embed_list.shape)
    np.savez_compressed(reduced_path, ids = ref_rid_list, embeds = ref_embed_list)