"""
data.py - Data loading, cleaning, and splitting for CDK2 QSAR/GNN models.
Fetches from ChEMBL on first run, caches locally for reproducibility.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_or_fetch_data(config):
    """
    Load cleaned CDK2 bioactivity data from cache or fetch from ChEMBL.
    
    Args:
        config: dict from config.yaml
    
    Returns:
        pd.DataFrame with columns [canonical_smiles, pchembl_value]
    """
    cache_path = config['data']['cache_path']
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        print(f"Loaded cached data: {len(df)} compounds from {cache_path}")
        return df
    
    print("Cache not found - fetching from ChEMBL (this takes 2-3 minutes)...")
    from chembl_webresource_client.new_client import new_client
    
    activity = new_client.activity
    res = activity.filter(
        target_chembl_id=config['data']['target_chembl_id'],
        standard_type='IC50',
        standard_relation='=',
        assay_type='B'
    ).only([
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_value',
        'standard_units',
        'pchembl_value',
        'assay_chembl_id'
    ])
    
    df_raw = pd.DataFrame(res)
    print(f"Raw records: {len(df_raw)}")
    
    # Clean
    df = df_raw.copy()
    df = df.dropna(subset=['canonical_smiles', 'pchembl_value'])
    df = df[df['standard_units'] == 'nM']
    df['pchembl_value'] = df['pchembl_value'].astype(float)
    df = df.groupby('canonical_smiles', as_index=False)['pchembl_value'].median()
    df = df[
        (df['pchembl_value'] >= config['data']['min_pic50']) &
        (df['pchembl_value'] <= config['data']['max_pic50'])
    ]
    
    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} compounds to {cache_path}")
    
    return df


def get_scaffold(smiles):
    """Get Murcko scaffold SMILES for a compound."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
    except:
        return None


def scaffold_split(df, test_size=0.2, random_seed=42):
    """
    Split dataset by Murcko scaffold - entire scaffold families
    assigned to train or test, never split across both.
    
    Args:
        df: DataFrame with canonical_smiles column
        test_size: fraction for test set
        random_seed: for reproducibility
    
    Returns:
        train_df, test_df
    """
    np.random.seed(random_seed)
    
    # Assign scaffolds
    scaffolds = defaultdict(list)
    for i, smi in enumerate(df['canonical_smiles']):
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            scaffolds[scaffold].append(i)
        else:
            scaffolds['no_scaffold'].append(i)
    
    # Sort by size descending for deterministic splits
    scaffold_sets = sorted(
        scaffolds.values(), 
        key=lambda x: len(x), 
        reverse=True
    )
    
    train_cutoff = int((1 - test_size) * len(df))
    train_idx, test_idx = [], []
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Scaffold split: {len(train_df)} train / {len(test_df)} test")
    print(f"Unique scaffolds: {len(scaffolds)}")
    
    return train_df, test_df


if __name__ == '__main__':
    # Quick test
    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    df = load_or_fetch_data(config)
    train_df, test_df = scaffold_split(df, config['data']['test_size'])
    
    print(f"\nTrain pIC50: {train_df.pchembl_value.mean():.2f} mean")
    print(f"Test pIC50:  {test_df.pchembl_value.mean():.2f} mean")