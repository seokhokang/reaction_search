import os, sys, csv
import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader
from dataset import ReactionDataset, collate_reaction, query_to_vec

from model import GINPredictor
from train import Trainer

from util import euclidean_sim, cosine_sim, is_valid_smiles

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--identifier', '-t', type=str, default='paper')
parser.add_argument('--scenario_id', '-s', type=int, default=1)
parser.add_argument('--batch_size', '-b', type=int, default=4096)
parser.add_argument('--frac_var', '-f', type=float, default=0.95)
parser.add_argument('--dim_reduction', '-r', type=int, default=1)

args = parser.parse_args()


# query selection
n_query = 10
n_retrieve = 30
n_updating = 3
random_state = 134


# configurations
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', cuda)
identifier = args.identifier
scenario_id = args.scenario_id
new_identifier = '%s_update_scenario_%d'%(identifier, scenario_id)
batch_size = args.batch_size
frac_var = args.frac_var
use_dim_reduction = (args.dim_reduction == 1)

with open('./data/uspto_mol_dict.pkl', 'rb') as f:
    mol_dict = pkl.load(f)

graph_net = GINPredictor(node_in_feats=140, edge_in_feats=8)


# user preference scenarios
def eval_ftn_1(r_product, r_reactant, query):

    q_product = Chem.MolFromSmiles(query['product'])
    r_product = Chem.MolFromSmiles(r_product)
    
    q_product_atoms = [atom.GetSymbol() for atom in q_product.GetAtoms()]
    r_product_atoms = [atom.GetSymbol() for atom in r_product.GetAtoms()]
    
    q_product_halogens = np.array([q_product_atoms.count(element) for element in ['F', 'Cl', 'Br', 'I', 'At']])
    r_product_halogens = np.array([r_product_atoms.count(element) for element in ['F', 'Cl', 'Br', 'I', 'At']])

    if np.array_equal(q_product_halogens, r_product_halogens):
        rating = 1
    else:
        rating = -1
        
    return rating
    
    
def eval_ftn_2(r_product, r_reactant, query):

    q_reactant_cnt = len(query['reactant'].split('.'))
    r_reactant_cnt = len(r_reactant.split('.'))

    if len(query['reactant']) > 0 and q_reactant_cnt == r_reactant_cnt:
        rating = 1
    elif len(query['reactant']) == 0 and r_reactant_cnt == 2:
        rating = 1
    else:
        rating = -1
        
    return rating


fpg = GetMorganGenerator()
def eval_ftn_3(r_product, r_reactant, query):

    r_product = Chem.MolFromSmiles(r_product)
    r_reactant_list = [Chem.MolFromSmiles(x) for x in r_reactant.split('.')]
    
    r_product_fp = fpg.GetFingerprint(r_product)
    r_reactant_fp_list = [fpg.GetFingerprint(x) for x in r_reactant_list]

    max_tanimoto_sim = np.max([TanimotoSimilarity(r_product_fp, x) for x in r_reactant_fp_list])
    if max_tanimoto_sim > 0.5:
        rating = 1
    else:
        rating = -1
        
    return rating


def eval_ftn_4(r_product, r_reactant, query):

    if eval_ftn_1(r_product, r_reactant, query) + eval_ftn_2(r_product, r_reactant, query) == 2:
        rating = 1
    else:
        rating = -1
        
    return rating
    

def eval_ftn_5(r_product, r_reactant, query):

    if eval_ftn_1(r_product, r_reactant, query) + eval_ftn_3(r_product, r_reactant, query) == 2:
        rating = 1
    else:
        rating = -1
        
    return rating  


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
    ref_embed_list = torch.FloatTensor(ref_embed_list)
    V = None

    # dimensionality reduction
    if use_dim_reduction:
        ref_cnt = ref_embed_list.shape[0]
        original_dim = ref_embed_list.shape[1] // 2

        ref_embed_list = torch.cat([ref_embed_list[:,:original_dim], ref_embed_list[:,original_dim:]], 0)
        
        _, S, V = torch.pca_lowrank(ref_embed_list, q=64, center=True, niter=2)
        exp_var_ratio = np.cumsum((S**2 / (len(ref_embed_list)-1)) / torch.var(ref_embed_list, dim=0).sum())
        reduced_dim = np.where(exp_var_ratio>frac_var)[0][0] + 1
    
        V = V[:,:reduced_dim]
        
        ref_embed_list = torch.matmul(ref_embed_list, V)
        ref_embed_list = torch.cat([ref_embed_list[:ref_cnt], ref_embed_list[ref_cnt:]], 1)

    return ref_rid_list, ref_embed_list, V


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

ref_data = ReactionDataset('uspto_train', mol_dict, mode = 'dict').data | ReactionDataset('uspto_valid', mol_dict, mode = 'dict').data


# query selection for search
query_data = ReactionDataset('uspto_test', mol_dict)

np.random.seed(random_state)
idx_list = np.random.permutation(len(query_data))[:n_query]
query_list = [query_data.reaction_list[idx] for idx in idx_list]

print('----------')
print('List of %d queries for search'%n_query)
for _query in query_list:

    query = {
        'Q.ID': _query[0],
        'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
        'reactant': '.'.join([mol_dict[x] for x in _query[2]]),#smiles
    }
    print(query)


eval_ftn = eval('eval_ftn_%d'%scenario_id)
result_list = []


# load the representation model
trainer = Trainer(graph_net, None, mol_dict, cuda)
trainer.load('./model/%s_checkpoint.pt'%identifier)

train_loader = DataLoader(
    dataset = ReactionDataset('uspto_train', mol_dict),
    batch_size = batch_size,
    collate_fn = collate_reaction,
    shuffle = True,
    drop_last = True
)


# chemical reaction search with user feedback (iteration of search and update)
retrieved_dict = {}
for iter_idx in range(n_updating+1):

    trainer.model_path  = './model/%s_iter_%d_checkpoint.pt'%(new_identifier, iter_idx + 1)
    
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
        ref_rid_list, ref_embed_list, V = get_embeddings()
    
    assert len(ref_data) == len(ref_embed_list)


    hit_ratio_list = []
    for q_idx, _query in enumerate(query_list):
    
        query_id = _query[0]
        if query_id not in retrieved_dict:
            retrieved_dict[query_id] = {}
        
        query = {
            'Q.ID': query_id,
            'product': '.'.join([mol_dict[x] for x in _query[1]]),#smiles
            'reactant': '.'.join([mol_dict[x] for x in _query[2]]),#smiles
        }
            
        # query processing
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
    
        # retrieve relevant records    
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
    
    # model update (assuming a single query)
    qvec_list = []
    rvec_list = []
    rating_list = []
    for query_id, query_retrieved_dict in retrieved_dict.items():
        query_retrieved_list = list(query_retrieved_dict.values())
        
        qvec_list.append(query_to_vec([query_id, query_retrieved_list[0][0], query_retrieved_list[0][1]]))
        rvec_list.append(collate_reaction([query_to_vec([query_id,v[2],v[3]]) for v in query_retrieved_list]))
        rating_list.append(torch.FloatTensor([v[4] for v in query_retrieved_list]))    

    trainer.update(qvec_list, rvec_list, rating_list, train_loader)        
        

for row in result_list:
    print(row)
        
