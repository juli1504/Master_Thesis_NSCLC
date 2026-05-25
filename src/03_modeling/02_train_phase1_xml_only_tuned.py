"""
Phase 1b: Global Baselines (Hyperparameter Tuned + TabPFN)

This script trains clinical baseline models (LR, MLP, XGB) using Optuna-optimized
hyperparameters and includes the state-of-the-art TabPFN for in-context learning.

Academic Rigor:
- Uses Train + Validation sets exclusively for fitting.
- Implements imblearn.Pipeline to prevent SMOTE data leakage during cross-validation.
- Evaluates the final models strictly on the untouched Test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def evaluate_model(name, model, X_test, y_test):
    # Note: TabPFN doesn't use predict_proba in the same way; adjusted for compatibility
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_probs = model.predict(X_test) # Fallback for models without predict_proba
        
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 0.5
        
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    
    return {
        "Model": name,
        "Accuracy": f"{acc * 100:.1f}%",
        "AUC": f"{auc:.3f}",
        "F1": f"{f1 * 100:.1f}%",
        "Sensitivity": f"{sensitivity * 100:.1f}%",
        "Specificity": f"{specificity * 100:.1f}%"
    }

def main():
    # --- 1. DATA LOADING & PREPROCESSING ---
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    df['target'] = LabelEncoder().fit_transform(df['histology'])
    
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_encoded = pd.get_dummies(df[clinical_features], columns=['Gender', 'Smoking status'], drop_first=True)
    
    train_val_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test'
    
    # Define the objects
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Fit and transform the Training/Validation set
    X_train_val_imputed = imputer.fit_transform(X_encoded[train_val_mask])
    X_train_val = scaler.fit_transform(X_train_val_imputed)
    
    # Use the ALREADY FITTED imputer and scaler to transform the Test set
    X_test_imputed = imputer.transform(X_encoded[test_mask])
    X_test = scaler.transform(X_test_imputed)
    
    y_train_val = df.loc[train_val_mask, 'target']
    y_test = df.loc[test_mask, 'target']
    
    # --- 2. MODELS DICTIONARY ---
    tabpfn_model = TabPFNClassifier(
        device='cpu', 
        model_path="03_modeling/tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )

    models = {
        "Tuned Logistic Regression": (
            ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))]),
            {'clf__C': [0.088987], 'clf__penalty': ['l2']}
        ),
        "Tuned MLP": (
            ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', MLPClassifier(max_iter=1000, random_state=42))]), 
            {'clf__hidden_layer_sizes': [(16,)], 'clf__learning_rate_init': [0.000287], 'clf__alpha': [0.000231]}
        ),
        "Tuned XGBoost": (
            ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', XGBClassifier(eval_metric='logloss', random_state=42))]), 
            {'clf__n_estimators': [121], 'clf__max_depth': [4], 'clf__learning_rate': [0.005335], 'clf__subsample': [0.666782], 'clf__colsample_bytree': [0.662423], 'clf__min_child_weight': [1]}
        ),
        "TabPFN": (
            tabpfn_model,
            {} 
        )
    }
    
    # --- 3. EXECUTION ---
    results = []
    print("\nEvaluating clinical baseline models on Test Set...")
    for name, (model, param_grid) in models.items():
        print(f"Fitting {name}...")
        if name == "TabPFN":
            # TabPFN is already instantiated and doesn't need GridSearch
            model.fit(X_train_val, y_train_val)
            results.append(evaluate_model(name, model, X_test, y_test))
        else:
            # Tuned models use the grid search
            gs = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train_val, y_train_val)
            results.append(evaluate_model(name, gs.best_estimator_, X_test, y_test))
        
    print("\n" + "="*80)
    print("PHASE 1b: FINAL CLINICAL BASELINE RESULTS (INCLUDING TABPFN)")
    print("="*80)
    print(pd.DataFrame(results).to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()