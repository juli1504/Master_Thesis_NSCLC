"""
Phase 1b: Unified Hyperparameter Optimization with Optuna
This script uses Bayesian Optimization to find the optimal hyperparameters for 
Logistic Regression, MLP, and XGBoost on the clinical data. 
It utilizes a rigorous SMOTE -> CV pipeline to mathematically prevent data leakage.
"""

import pandas as pd
import numpy as np
import optuna
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Make Optuna output slightly less verbose so we can read the final results easily
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def load_and_preprocess_data():
    """Loads data, applies encoding, and returns strictly the Training Set."""
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    # Strictly isolate the Training set
    train_mask = df['dataset_split'] == 'Train'
    X_train_raw = X_encoded[train_mask]
    y_train = df.loc[train_mask, 'target']
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train_imputed)
    
    return X_train, y_train

# --- 2. OPTUNA OBJECTIVE FUNCTIONS ---

def objective_lr(trial, X_train, y_train, cv):
    """Search space for Logistic Regression"""
    param = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'liblinear', # liblinear supports both l1 and l2
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(**param))
    ])
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc').mean()

def objective_mlp(trial, X_train, y_train, cv):
    """Search space for Multi-Layer Perceptron"""
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(16,), (32, 16), (64, 32), (128, 64, 32)]),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True), # L2 Regularization
        'max_iter': 1000,
        'random_state': 42
    }
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', MLPClassifier(**param))
    ])
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc').mean()

def objective_xgb(trial, X_train, y_train, cv):
    """Search space for XGBoost"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(**param))
    ])
    return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc').mean()

# --- 3. MAIN EXECUTION ---
def main():
    print("Loading data and setting up Stratified CV...")
    X_train, y_train = load_and_preprocess_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}

    # 1. Tune Logistic Regression
    print("\n--- Tuning Logistic Regression (50 trials) ---")
    study_lr = optuna.create_study(direction='maximize', study_name="LR_Tuning")
    study_lr.optimize(lambda trial: objective_lr(trial, X_train, y_train, cv), n_trials=50)
    results["Logistic Regression"] = study_lr
    print(f"Best CV AUC: {study_lr.best_value:.3f}")

    # 2. Tune MLP
    print("\n--- Tuning MLP Neural Network (50 trials) ---")
    study_mlp = optuna.create_study(direction='maximize', study_name="MLP_Tuning")
    study_mlp.optimize(lambda trial: objective_mlp(trial, X_train, y_train, cv), n_trials=50)
    results["MLP (Neural Net)"] = study_mlp
    print(f"Best CV AUC: {study_mlp.best_value:.3f}")

    # 3. Tune XGBoost
    print("\n--- Tuning XGBoost (50 trials) ---")
    study_xgb = optuna.create_study(direction='maximize', study_name="XGB_Tuning")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train, cv), n_trials=50)
    results["XGBoost"] = study_xgb
    print(f"Best CV AUC: {study_xgb.best_value:.3f}")

    # --- 4. PRINT FINAL COPY-PASTE READY RESULTS ---
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE! COPY THESE INTO YOUR PHASE 1B SCRIPT:")
    print("="*70)
    for model_name, study in results.items():
        print(f"\n{model_name} Best Params:")
        for key, value in study.best_trial.params.items():
            if isinstance(value, float):
                print(f"    'clf__{key}': [{value:.6f}],")
            elif isinstance(value, str):
                print(f"    'clf__{key}': ['{value}'],")
            else:
                print(f"    'clf__{key}': [{value}],")
    print("="*70)

if __name__ == "__main__":
    main()