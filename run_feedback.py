"""Chemical reaction search with user feedback. 
 It allows for interaction through user-defined scenarios to refine the search based on feedback.

This script:
- Loads a reaction dataset and a GIN-based representation model.
- Performs reaction search with user feedback across multiple iterations for given queries.
- Updates the model iteratively based on user feedback to enhance search results.
- Evaluates chemical reaction search performance.

Usage example (identifier: 'paper', dim_reduction: True, scenario_id: 1):
    python run_feedback.py -t paper -r 1 -s 1
"""

import os, sys, csv, yaml
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction, query_to_vec
from train import Trainer
from util import euclidean_sim, is_valid_smiles, get_embeddings
from scenario import *
from argparse import ArgumentParser

# ------------------------------
# Configurations for Query Selection and Search
# ------------------------------
n_query = 10
product_only = False
n_retrieve = 30
n_updating = 3
random_state = 134

# ------------------------------
# Configurations and Setup
# ------------------------------
config_url = 'config.yaml'
with open(config_url, 'r') as f:
    config = yaml.safe_load(f)  

parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--dim_reduction', '-r', type=int, default=1)
parser.add_argument('--scenario_id', '-s', type=int, default=1)
args = parser.parse_args()

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
use_dim_reduction = (args.dim_reduction == 1)
scenario_id = args.scenario_id
eval_ftn = eval('eval_ftn_%d'%scenario_id)
new_identifier = '%s_%s_update_scenario_%d'%(identifier, 'product' if product_only else 'reaction', scenario_id)

temp_tau = config['train']['tau']
batch_size = config['train']['batch_size']
frac_var = config['pca']['frac_var']

# Load molecular dictionary
with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

ref_data = ReactionDataset('uspto_train', mol_dict, mode = 'dict').data | ReactionDataset('uspto_valid', mol_dict, mode = 'dict').data

# ------------------------------
# Query Selection for Search
# ------------------------------
query_data = ReactionDataset('uspto_test', mol_dict)

np.random.seed(random_state)
idx_list = np.random.permutation(len(query_data))[:n_query]
query_list = [query_data.reaction_list[idx] for idx in idx_list]

print('----------')
print('List of %d queries for search'%n_query)
for _query in query_list:

    if product_only:
        query = {
            'Q.ID': _query[0],
            'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
            'reactant': ''
        }
    else:
        query = {
            'Q.ID': _query[0],
            'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
            'reactant': '.'.join([mol_dict[x] for x in _query[2]])#smiles
        }
    print(query)
print('----------')

# Load the representation model
trainer = Trainer(None, None, mol_dict, cuda, tau = temp_tau)
trainer.load('./model/%s_checkpoint.pt'%identifier)

train_loader = DataLoader(
    dataset = ReactionDataset('uspto_train', mol_dict),
    batch_size = batch_size,
    collate_fn = collate_reaction,
    shuffle = True,
    drop_last = True
)

# ------------------------------
# Chemical Reaction Search with User Feedback (iteration of search and update)
# ------------------------------
result_list = []
retrieved_dict = {}
for iter_idx in range(n_updating+1):

    trainer.save_path = './model/%s_iter_%d_checkpoint.pt'%(new_identifier, iter_idx + 1)
    
    if iter_idx == 0 and not use_dim_reduction and os.path.exists('./embed/%s_embeddings.npz'%identifier):
        data = np.load('./embed/%s_embeddings.npz'%identifier)
        ref_rid_list = data['ids']
        ref_embed_list = torch.FloatTensor(data['embeds'])
        
    elif iter_idx == 0 and use_dim_reduction and os.path.exists('./embed/%s_embeddings_reduced.npz'%identifier):
        pca = np.load('./embed/%s_pca.npz'%identifier)
        V = torch.FloatTensor(pca['pc'])
        
        data = np.load('./embed/%s_embeddings_reduced.npz'%identifier)
        ref_rid_list = data['ids']
        ref_embed_list = torch.FloatTensor(data['embeds'])

    else:
        ref_rid_list, ref_embed_list, V = get_embeddings(trainer, use_dim_reduction, frac_var)
    
    assert len(ref_data) == len(ref_embed_list)

    # Reaction search and update based on feedback
    hit_ratio_list = []
    for q_idx, _query in enumerate(query_list):
    
        # Query processing
        query_id = _query[0]
        if query_id not in retrieved_dict:
            retrieved_dict[query_id] = {}
        
        if product_only:
            query = {
                'Q.ID': _query[0],
                'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
                'reactant': ''
            }
        else:
            query = {
                'Q.ID': _query[0],
                'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
                'reactant': '.'.join([mol_dict[x] for x in _query[2]])#smiles
            }

        has_product = is_valid_smiles(query['product'])
        has_reactant = is_valid_smiles(query['reactant']) 
        if len(query['product']) > 0: assert has_product
        if len(query['reactant']) > 0: assert has_reactant
        assert has_product or has_reactant
        
        print('----------') 
        print('QUERY %d = %s'%(q_idx, query), has_product, has_reactant)
        #print('----------') 
        
        q_vec = query_to_vec(query)

        q_product, q_product_pred = trainer.embed(q_vec[1], q_vec[2], to_numpy = False)
        if not has_reactant: q_product_pred = q_product
        if not has_product: q_product = q_product_pred
        assert torch.sum(torch.abs(q_product)) > 0 and torch.sum(torch.abs(q_product_pred)) > 0 
        if use_dim_reduction:
            q_product = torch.matmul(q_product, V)
            q_product_pred = torch.matmul(q_product_pred, V)
    
        # Retrieve relevant records    
        q_embed = torch.cat([q_product, q_product_pred], dim=1)
        sim = euclidean_sim(q_embed, ref_embed_list).numpy().ravel()
        sort_idx = np.argsort(-sim)
    
        duplicate_check = []
        iter_score = []
        for i, idx in enumerate(sort_idx):
            rid = ref_rid_list[idx]
            inst = ref_data[rid]
        
            res_product = mol_dict.get(inst['product'][0])
            res_reactant = '.'.join([mol_dict.get(x) for x in inst['reactant']])
            key = '%s_%s_%s'%(query_id, res_product, res_reactant) 
            if key in duplicate_check:
                #print(key, 'duplicate, skip')
                continue
            else:
                duplicate_check.append(key)
            
            rating = eval_ftn(res_product, res_reactant, query)
            retrieved_dict[query_id][key] = [query['product'], query['reactant'], res_product, res_reactant, rating]
    
            iter_score.append(rating)
        
            #print('RANK %d; RX.ID %s; {product: %s, reactant: %s}; dist %.3f; rating %d'%(len(iter_score), rid, res_product, res_reactant, np.sqrt(-sim[idx]), rating))
        
            if len(iter_score) == n_retrieve:
                break  
        
        hit_ratio = 100 * (np.mean(iter_score) + 1) / 2
        hit_ratio_list.append(hit_ratio)
    
        #print('----------')
        print('QUERY_ID %s, NO. UPDATES = %d, HIT RATIO = %.2f percent, R_CNT %d'%(query_id, iter_idx, hit_ratio, len(retrieved_dict[query_id])))

    print('----------')
    result_list.append([scenario_id, iter_idx] + hit_ratio_list)
    if iter_idx == n_updating: break
    
    # Model update
    qvec_list = []
    rvec_list = []
    rating_list = []
    for query_id, query_retrieved_dict in retrieved_dict.items():
        query_retrieved_list = list(query_retrieved_dict.values())
        
        qvec_list.append(query_to_vec([query_id, query_retrieved_list[0][0], query_retrieved_list[0][1]]))
        rvec_list.append(collate_reaction([query_to_vec([query_id,v[2],v[3]]) for v in query_retrieved_list]))
        rating_list.append(torch.FloatTensor([v[4] for v in query_retrieved_list]))    

    trainer.update(qvec_list, rvec_list, rating_list, train_loader)        
        

with open('%s.csv'%new_identifier, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in result_list:
        writer.writerow(row)
        print(row)       
