import os, sys
import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction

from model import GINPredictor
from train import Trainer

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--batch_size', '-b', type=int, default=4096)
parser.add_argument('--frac_var', '-f', type=float, default=0.95)

args = parser.parse_args()


# configurations
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
batch_size = args.batch_size
frac_var = args.frac_var

if not os.path.exists('./model/'): os.makedirs('./model/')
if not os.path.exists('./embed/'): os.makedirs('./embed/')

with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

graph_net = GINPredictor(node_in_feats=140, edge_in_feats=8)


# train the representation model
model_path = './model/%s_checkpoint.pt'%identifier
trainer = Trainer(graph_net, model_path, mol_dict, cuda)

print('train model')

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


# extract embeddings
embed_path = './embed/%s_embeddings.npz'%identifier

print('extract embeddings')

ref_rid_list = []
ref_embed_list = []

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

for loader in [ref_train_loader, ref_valid_loader]:
    for batch_idx, batch_data in enumerate(loader):
        rid = np.array(batch_data[0])
        product, product_pred = trainer.embed(batch_data[1], batch_data[2])
        embed = np.concatenate([product, product_pred], axis=1)

        ref_rid_list.extend(rid)
        ref_embed_list.append(embed)

ref_rid_list = np.array(ref_rid_list)        
ref_embed_list = np.vstack(ref_embed_list).astype(np.float16)
    
print(ref_rid_list.shape, ref_embed_list.shape)
np.savez_compressed(embed_path, ids = ref_rid_list, embeds = ref_embed_list)


# dimensionality reduction
reduced_path = './embed/%s_embeddings_reduced.npz'%identifier
pca_path = './embed/%s_pca.npz'%identifier

print('reduce dimensionality')

ref_cnt = ref_embed_list.shape[0]
original_dim = ref_embed_list.shape[1] // 2

ref_embed_list = torch.FloatTensor(ref_embed_list)
ref_embed_list = torch.cat([ref_embed_list[:,:original_dim], ref_embed_list[:,original_dim:]], 0)

_, S, V = torch.pca_lowrank(ref_embed_list, q=64, center=True, niter=2)
exp_var_ratio = np.cumsum((S**2 / (len(ref_embed_list)-1)) / torch.var(ref_embed_list, dim=0).sum())
reduced_dim = np.where(exp_var_ratio>frac_var)[0][0] + 1
print(exp_var_ratio)
print(reduced_dim)
print(exp_var_ratio[:reduced_dim])

V = V[:,:reduced_dim]

ref_embed_list = torch.matmul(ref_embed_list, V)
ref_embed_list = torch.cat([ref_embed_list[:ref_cnt], ref_embed_list[ref_cnt:]], 1)
ref_embed_list = ref_embed_list.cpu().numpy().astype(np.float16)

V = V.cpu().numpy()

print(V.shape)
np.savez_compressed(pca_path, pc = V)

print(ref_rid_list.shape, ref_embed_list.shape)
np.savez_compressed(reduced_path, ids = ref_rid_list, embeds = ref_embed_list)