"""
Phase 1b: Hyperparameter Optimization with Optuna (XGBoost)
This script uses Optuna to find the optimal hyperparameters for XGBoost 
on the clinical data, using a rigorous SMOTE -> CV pipeline to prevent leakage.
"""

import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- 1. CONFIGURATION (Same as your Phase 1 script) ---
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
    
    # We strictly only tune on the Training set!
    train_mask = df['dataset_split'] == 'Train'
    X_train_raw = X_encoded[train_mask]
    y_train = df.loc[train_mask, 'target']
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train_imputed)
    
    return X_train, y_train

def objective(trial):
    """Optuna objective function: defines search space and evaluates model."""
    X_train, y_train = load_and_preprocess_data()
    
    # 1. Define the Search Space (This is what you report in your Appendix!)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'random_state': 42
    }
    
    # 2. Setup the strict SMOTE -> Model Pipeline
    model = XGBClassifier(**param)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # 3. 5-Fold Cross Validation optimizing for AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

def main():
    print("Starting Optuna Hyperparameter Optimization for XGBoost...")
    
    # Create the study and tell it to maximize our metric (AUC)
    study = optuna.create_study(direction='maximize', study_name="XGBoost_Clinical_Tuning")
    
    # Run 50 experiments (trials)
    study.optimize(objective, n_trials=50)
    
    print("\n" + "="*50)
    print("OPTIMIZATION FINISHED!")
    print(f"Best CV AUC Score: {study.best_value:.3f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    '{key}': {value}")
    print("="*50)
    print("Action: Take these best parameters and paste them into your Phase 1b script!")

if __name__ == "__main__":
    main()