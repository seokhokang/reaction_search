# reaction_search
Pytorch implementation of the method described in the paper [Enhancing Chemical Reaction Search through Contrastive Representation Learning and Human-in-the-Loop](#)

## Components
- **data/*** - data files used
- **data/uspto479k.py** - script for preprocessing the USPTO-479k dataset
- **dataset.py** - data structure & functions
- **featurizer.py** - featurization of nodes and edges in of molecular graphs
- **model.py** - model architecture
- **train.py** - model training/inference functions
- **util.py**
- **run_train.py** - script for model training (representation learning)
- **run_product_prediction.py** - script for reaction product prediction
- **run_search.py** - script for chemical reaction search
- **run_feedback.py** - script for chemical reaction search with user feedback (5 user preference scenarios described in the paper)

## Data
- The USPTO-479k dataset can be downloaded from
  - https://github.com/hwwang55/MolR/tree/master/data/USPTO-479k

## Usage Example
`python run_train.py`
`python run_product_prediction.py`
`python run_feedback.py -s 1`

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation

