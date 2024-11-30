import os, sys
import numpy as np
import torch
import dgl
import pickle as pkl

from featurizer import *

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')


class ReactionDataset():

    def __init__(self, filename, mol_dict, mode = 'reaction'):
 
        assert mode in ['reaction','product','dict']
 
        self.mol_dict = mol_dict
    
        with open('./data/%s.pkl'%filename, 'rb') as f:
            self.data = pkl.load(f)
        
        self.mode = mode
        print('imported dataset ./data/%s.pkl'%filename)

        if mode == 'reaction':
            self.reaction_list = []
            for k, v in self.data.items():
                self.reaction_list.append([k, v['product'], v['reactant']])
            print('preprocessed %d reaction records'%len(self.reaction_list))
              
        elif mode == 'product':
            self.product_list = []
            for k, v in self.data.items():
                self.product_list.extend(v['product'])
            self.product_list = list(set(self.product_list))
            print('preprocessed %d product records'%len(self.product_list))


    def __getitem__(self, idx):

        if self.mode == 'reaction':
            [rid, product, reactant] = self.reaction_list[idx]
            product = smi_to_graph('.'.join([self.mol_dict.get(x) for x in product]), product)
            reactant = smi_to_graph('.'.join([self.mol_dict.get(x) for x in reactant]), reactant)

            return [rid, product, reactant]

        elif self.mode == 'product':
            product_id = self.product_list[idx]
            product = smi_to_graph(self.mol_dict.get(product_id), product_id)
            
            return [product_id, product]
        
        
    def __len__(self):
    
        if self.mode == 'reaction':
            return len(self.reaction_list)
            
        elif self.mode == 'product':
            return len(self.product_list)


def query_to_vec(query):

    """    
    query = {
        'Q.ID': 1234,
        'product': '',#smiles
        'reactant': 'CC(=O)OC(C)=O',#smiles
    }
    
    or
    
    query = [1234, 'CC', 'NN']
    """

    if isinstance(query, dict):
        qid = query['Q.ID']
        product = query['product']
        reactant = query['reactant']
        
    elif isinstance(query, list):
        qid = query[0]
        product = query[1]
        reactant = query[2]
    
    product = smi_to_graph(product)
    reactant = smi_to_graph(reactant)

    return [qid, product, reactant]
    
        
def collate_reaction(batch):

    batchdata = list(map(list, zip(*batch)))
    rid = batchdata[0]
    gs = [dgl.batch(s) for s in batchdata[1:]]
    
    return rid, *gs
    
    
def collate_product(batch):

    return collate_reaction(batch)