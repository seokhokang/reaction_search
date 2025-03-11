"""Functions to evaluate chemical reactions based on user preference scenarios.
It returns 1 if the given condition for a positive rating is satisfied, and -1 otherwise.

Functions:
    eval_ftn_1: Checks if the product in the retrieved reaction contains the same halogen atoms as the product in the query.
    eval_ftn_2: Compares the number of reactants in the retrieved reaction with the number of reactants in the query (or checks if there are 2 reactants when none are specified in the query).
    eval_ftn_3: Determines if the maximum Tanimoto similarity between the product and any reactant in the retrieved reaction exceeds 0.5.
    eval_ftn_4: Verifies that both conditions in scenarios 1 and 2 are satisfied in the retrieved reaction.
    eval_ftn_5: Verifies that both conditions in scenarios 1 and 3 are satisfied in the retrieved reaction.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity


def eval_ftn_1(r_product, r_reactant, query):
    """Evaluates if the halogen atom count in the product matches the query.

    Args:
        r_product (str): SMILES string of the product molecule.
        r_reactant (str): SMILES string of the reactant molecule (unused).
        query (dict): Dictionary containing 'product' SMILES string.

    Returns:
        int: 1 if the product in the retrieved reaction contains the same halogen atoms as the product in the query, -1 otherwise.
    """
    
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
    """Compares the count of reactants in the reaction to a query.

    Args:
        r_product (str): SMILES string of the product molecule (unused).
        r_reactant (str): SMILES string of all reactants.
        query (dict): Dictionary containing 'reactant' SMILES string.

    Returns:
        int: 1 if the number of reactants in the retrieved reaction matches the number of reactants in the query (or is 2 if no reactants are specified in the query), -1 otherwise.
    """

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
    """Computes the maximum Tanimoto similarity between the product and reactants' fingerprints.

    Args:
        r_product (str): SMILES string of the product molecule.
        r_reactant (str): SMILES string of all reactants, separated by dots.
        query (dict): Dictionary containing query details (unused).

    Returns:
        int: 1 if the maximum Tanimoto similarity between the product and any reactant in the retrieved reaction exceeds 0.5, -1 otherwise.
    """

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
    """Aggregates evaluations from functions 1 and 2 to compute a combined rating.

    Args:
        r_product (str): SMILES string of the product molecule.
        r_reactant (str): SMILES string of the reactant molecule.
        query (dict): Dictionary containing query details.

    Returns:
        int: 1 if both function evaluations are positive, -1 otherwise.
    """

    if eval_ftn_1(r_product, r_reactant, query) + eval_ftn_2(r_product, r_reactant, query) == 2:
        rating = 1
    else:
        rating = -1
        
    return rating
    

def eval_ftn_5(r_product, r_reactant, query):
    """Aggregates evaluations from functions 1 and 3 to compute a combined rating.

    Args:
        r_product (str): SMILES string of the product molecule.
        r_reactant (str): SMILES string of the reactant molecule.
        query (dict): Dictionary containing query details.

    Returns:
        int: 1 if both function evaluations are positive, -1 otherwise.
    """

    if eval_ftn_1(r_product, r_reactant, query) + eval_ftn_3(r_product, r_reactant, query) == 2:
        rating = 1
    else:
        rating = -1
        
    return rating  