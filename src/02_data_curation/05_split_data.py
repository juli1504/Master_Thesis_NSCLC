"""
Dataset Splitting Script.

This script adds a 'dataset_split' column to the manifest, stratifying 
by histology to ensure an equal distribution of cancer subtypes across 
the training, validation, and test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

def main():
    print("Creating stratified data split...")
    
    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    
    # Filter for patients that passed QC and have patches
    valid_df = df[df['patch_extracted'] == True].copy()
    
    # 1. Split into Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        valid_df, 
        test_size=0.30, 
        stratify=valid_df['histology'], 
        random_state=42 # Fixed seed for reproducibility
    )
    
    # 2. Split Temp into Validation (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['histology'], 
        random_state=42
    )
    
    # 3. Assign labels back to the main dataframe
    df['dataset_split'] = 'Excluded'
    df.loc[train_df.index, 'dataset_split'] = 'Train'
    df.loc[val_df.index, 'dataset_split'] = 'Validation'
    df.loc[test_df.index, 'dataset_split'] = 'Test'
    
    # Save the updated manifest
    df.to_csv(FILE_MANIFEST, index=False, sep=';', decimal=',')
    
    print("Split complete. Distribution:")
    print(df['dataset_split'].value_counts())

if __name__ == "__main__":
    main()