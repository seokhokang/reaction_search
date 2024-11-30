import pickle as pkl
import numpy as np
import pandas as pd

## The USPTO-479k dataset can be downloaded from
## https://github.com/hwwang55/MolR/tree/master/data/USPTO-479k

mol_id = 0
mol_dict_inv = {}

for split in ['train', 'valid', 'test']:

    csvdata = pd.read_csv('./USPTO-479k/%s.csv'%split, header = 0).to_numpy()

    data = {}
    for ins in csvdata:
    
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
        
        data[key] = {
            'product': product_list,
            'reactant': reactant_list
        }
    
    print(split, len(data))
    with open('uspto_%s.pkl'%split, 'wb') as f:
        pkl.dump(data, f)
        
mol_dict = {}
for k, v in mol_dict_inv.items():
    mol_dict[v] = k
        
with open('uspto_mol_dict.pkl', 'wb') as f:
    pkl.dump(mol_dict, f)