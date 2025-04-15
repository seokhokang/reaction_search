# reaction_search
Pytorch implementation of the method described in the paper [Enhancing Chemical Reaction Search through Contrastive Representation Learning and Human-in-the-Loop](#)

## Overview
- This study introduces an advanced chemical reaction search mechanism that refines search results based on user input. It employs contrastive representation learning to train models that embed chemical reactions as numerical vectors, facilitating efficient and relevant searches. Dimensionality reduction techniques are applied to optimize these vectors for quicker processing, while human-in-the-loop integration allows for continuous improvement of the model based on user feedback.
- This GitHub repository provides running examples of the proposed method on the USPTO-479k dataset.

## Components
- **data/*** - Directory for data files.
- **data/uspto479k.py** - Script for preprocessing the USPTO-479k dataset.
- **model/*** - Directory for pretrained models.
- **embed/*** - Directory for saved embeddings.
- **dataset.py** - Defines data structures and functions for dataset operations.
- **featurizer.py** - Functions for the featurization of nodes and edges in molecular graphs.
- **model.py** - Defines the neural network model architecture.
- **train.py** -  Classes and functions for model training and inference.
- **scenario.py** - The five user preference scenarios described in the paper.
- **util.py** - Functions used across scripts.
- **run_train.py** - Script for model training (representation learning)
- **run_product_prediction.py** - Script for reaction product prediction
- **run_feedback.py** - Script for chemical reaction search with user feedback
- **config.yaml** - Default configuration file for running scripts.
- **requirements.txt** - Package dependencies required.
- **Reaction_Search_Example.ipynb** - Jupyter Notebook example for chemical reaction search using a pretrained model.


## Usage Example

### Data download and processing
- The USPTO-479k dataset can be downloaded from
  - https://github.com/hwwang55/MolR/tree/master/data/USPTO-479k
- After downloading the dataset and placing it in the `./data/` directory, preprocess it by running the following command:
```python
python ./data/uspto479k.py
```

### Training a representation model
- To train the representation model on the USPTO-479k dataset, run the following command:
```python
python run_train.py
```
- The trained model is stored in the `./model/` directory.
- Data embeddings and principal components are stored in `./embed/` directory.

### Reaction product prediction
- To perform reaction product prediction, run the following command:
```python
python run_product_prediction.py
```

### Chemical reaction search with user feedback
- To perform a chemical reaction search with user feedback,
  first select from preference scenarios 1-5 
  or manually define the scenario to positively/negatively rate each retrieved record.
- For example, to use scenario 1, run the following command:
```python
python run_feedback.py -s 1
```

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Citation
```
@Article{Kwon2025,
  title={Enhancing chemical reaction search through contrastive representation learning and human-in-the-loop},
  author={Kwon, Youngchun and Jeon, Hyunjung and Choi, Joonhyuk and Choi, Youn-Suk and Kang, Seokho},
  journal={Journal of Cheminformatics},
  volume={17},
  pages={51},
  year={2025},
  doi={10.1186/s13321-025-00987-5}
}
```
