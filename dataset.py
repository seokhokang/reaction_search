"""Class and functions for handling chemical reaction data for graph-based rrepresentation learning.

Key Features:
- Loading and preprocessing reaction data into molecular graph representations.
- Providing utilities for converting queries into structured formats with molecular graphs.
- Collating batches of data for efficient processing in graph neural networks.

Class:
- ReactionDataset: Manages reaction data loading and preprocessing.

Functions:
- query_to_vec: Converts queries into molecular graph formats.
- collate_reaction: Prepares batches of reaction data for DGL processing.
- collate_product: Prepares batches of product data, similar to collate_reaction.
"""

import os, sys
import numpy as np
import torch
import dgl
import pickle as pkl

from featurizer import *

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')


class ReactionDataset():
    """A dataset class for handling reaction data.

    This class loads reaction data from a pickle file and preprocesses it into a format 
    suitable for graph-based reaction representation learning.
    """

    def __init__(self, filename, mol_dict, mode = 'reaction'):
        """Initializes the ReactionDataset class and loads the dataset.

        Args:
            filename (str): Name of the dataset file.
            mol_dict (dict): Dictionary mapping molecular indices to SMILES.
            mode (str, optional): Dataset mode. Can be 'reaction' or 'product'. Default is 'reaction'.
        """

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
        """Retrieves a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            list: A list containing reaction ID and corresponding molecular graphs.
                  - If mode is 'reaction': [reaction_id, product_graph, reactant_graph]
                  - If mode is 'product': [product_id, product_graph]
        """

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
        """Returns the length of the dataset.

        Returns:
            int: Number of records in the dataset.
        """
    
        if self.mode == 'reaction':
            return len(self.reaction_list)
            
        elif self.mode == 'product':
            return len(self.product_list)


def query_to_vec(query):
    """Converts a query dictionary or list into a structured format with molecular graphs.

    Args:
        query (dict or list): Query information containing reaction details.
            - If a dictionary, it should have keys: 'Q.ID', 'product', and 'reactant'.
            - If a list, it should contain: [Q.ID, product SMILES, reactant SMILES].

    Returns:
        list: [Q.ID, product_graph, reactant_graph]
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
    """Collates a batch of reaction data into a format suitable for DGL processing.

    Args:
        batch (list): List of reaction data instances.

    Returns:
        tuple: (reaction_ids, batched_product_graphs, batched_reactant_graphs)
    """

    batchdata = list(map(list, zip(*batch)))
    rid = batchdata[0]
    gs = [dgl.batch(s) for s in batchdata[1:]]
    
    return rid, *gs
    
    
def collate_product(batch):
    """Collates a batch of product data into a format suitable for DGL processing.

    Args:
        batch (list): List of product data instances.

    Returns:
        tuple: Collated product data, same as `collate_reaction`.
    """

    return collate_reaction(batch)