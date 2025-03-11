"""This script processes the USPTO-479k dataset and saves it into separate files for training, validation, and testing.
The dataset can be obtained from:  https://github.com/hwwang55/MolR/tree/master/data/USPTO-479k
"""

import pickle as pkl
import numpy as np
import pandas as pd


# Initialize molecule ID and inverse dictionary to keep track of SMILES strings and their assigned indices
mol_id = 0
mol_dict_inv = {}

for split in ['train', 'valid', 'test']:

    csvdata = pd.read_csv('./USPTO-479k/%s.csv'%split, header = 0).to_numpy()

    data = {}
    for ins in csvdata:
        # Construct a unique key for each entry combining the split and the first column from the CSV
        key = '%s_%d'%(split, ins[0])    
        product_smi = ins[1].split('.')
        reactant_smi = ins[2].split('.')
    
        product_list = []
        for smi in product_smi:
            if smi not in mol_dict_inv:
                mol_dict_inv[smi] = mol_id
                mol_id += 1
                
            product_list.append(mol_dict_inv[smi])  
            
        reactant_list = []
        for smi in reactant_smi:
            if smi not in mol_dict_inv:
                mol_dict_inv[smi] = mol_id
                mol_id += 1
                
            reactant_list.append(mol_dict_inv[smi])      
        
        # Store product and reactant indices in a dictionary under their respective keys
        data[key] = {
            'product': product_list,
            'reactant': reactant_list
        }
    
    print(split, len(data))
    
    # Save the processed data into a pickle file
    with open('uspto_%s.pkl'%split, 'wb') as f:
        pkl.dump(data, f)

# Create a dictionary that maps molecule indices back to SMILES strings        
mol_dict = {}
for k, v in mol_dict_inv.items():
    mol_dict[v] = k

# Save the molecule dictionary as a pickle file
with open('uspto_mol_dict.pkl', 'wb') as f:
    pkl.dump(mol_dict, f)