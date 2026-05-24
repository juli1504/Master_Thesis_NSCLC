"""
Phase 1b: Global Baselines (Hyperparameter Tuned with Bounded Dynamic Expansion)

This script uses GridSearchCV to automatically experiment with different
hyperparameters for MLP and XGBoost. If a best parameter hits the edge 
of a provided numeric grid (left OR right), it dynamically expands the search space!

Academic Rigor:
- Uses Train + Validation sets exclusively for the tuning (GridSearchCV).
- Implements imblearn.Pipeline to prevent SMOTE data leakage during cross-validation.
- Evaluates the final ultimate models strictly on the untouched Test set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore') # Hides annoying deprecation warnings from Sklearn

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def evaluate_model(name, model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_probs)
    except ValueError:
        auc = 0.500
        
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)  # Macro-averaged F1 by default for binary classification
    
    return {
        "Model": name,
        "Accuracy": f"{acc * 100:.1f}%",
        "AUC": f"{auc:.3f}",
        "F1": f"{f1 * 100:.1f}%",  # Added F1 score
        "Sensitivity": f"{sensitivity * 100:.1f}%",
        "Specificity": f"{specificity * 100:.1f}%"
    }

def dynamic_grid_search(model, param_grid, X, y, cv=5, max_expansions=3):
    """
    Runs a GridSearchCV with Smart Multipliers and Mathematical Guardrails!
    """
    current_grid = param_grid.copy()

    # Hard limits for ALL parameters so the expander doesn't break algorithms
    param_bounds = {
        'clf__subsample': (0.1, 1.0),
        'clf__learning_rate': (0.00001, 1.0),
        'clf__learning_rate_init': (0.00001, 1.0),
        'clf__max_depth': (1, 50),
        'clf__n_estimators': (10, 5000),
        'clf__C': (0.0001, 10000.0),
        'clf__alpha': (0.000001, 10.0)
    }

    for attempt in range(max_expansions + 1):
        gs = GridSearchCV(model, current_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        gs.fit(X, y)
        best_params = gs.best_params_

        expanded = False
        new_grid = {}

        for param, values in current_grid.items():
            if isinstance(values, list) and len(values) > 1 and isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                sorted_vals = sorted(values)
                best_val = best_params[param]
                
                min_bound, max_bound = param_bounds.get(param, (0.000001, float('inf')))

                # --- UPPER EDGE HIT (Expanding to the Right) ---
                if best_val == sorted_vals[-1]:
                    step = sorted_vals[-1] - sorted_vals[-2]
                    new_val = best_val + step
                    
                    if sorted_vals[-2] > 0 and (sorted_vals[-1] / sorted_vals[-2] >= 2.0):
                        new_val = best_val * (sorted_vals[-1] / sorted_vals[-2])
                        
                    if isinstance(best_val, int): new_val = int(round(new_val))
                    else: new_val = round(new_val, 6)
                    
                    if new_val <= max_bound and new_val != best_val:
                        new_grid[param] = sorted_vals + [new_val]
                        expanded = True
                    else:
                        new_grid[param] = values
                
                # --- LOWER EDGE HIT (Expanding to the Left) ---
                elif best_val == sorted_vals[0]:
                    step = sorted_vals[1] - sorted_vals[0]
                    new_val = best_val - step
                    
                    if sorted_vals[0] > 0:
                        ratio = sorted_vals[1] / sorted_vals[0]
                        if ratio >= 2.0 or new_val < min_bound:
                            new_val = best_val / ratio
                            
                    if isinstance(best_val, int): new_val = int(round(new_val))
                    else: new_val = round(new_val, 6)
                    
                    if new_val >= min_bound and new_val != best_val:
                        new_grid[param] = [new_val] + sorted_vals
                        expanded = True
                    else:
                        new_grid[param] = values
                else:
                    new_grid[param] = values
            else:
                new_grid[param] = values

        if expanded and attempt < max_expansions:
            print(f"Edge hit detected! Expanding grid space to: {new_grid}")
            current_grid = new_grid
        else:
            return gs

def main():
    print("Loading data for Hyperparameter Tuning...\n")
    
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    # --- STRICT HOLDOUT SPLIT ---
    # Combine Train & Val for the Tuning phase
    train_val_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test' # Locked strictly in the vault
    
    X_train_val_raw = X_encoded[train_val_mask]
    y_train_val = df.loc[train_val_mask, 'target']
    
    X_test_raw = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target']
    
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    
    # Impute and Scale based on the tuning data, apply to test data
    X_train_val_imputed = imputer.fit_transform(X_train_val_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    
    X_train_val = scaler.fit_transform(X_train_val_imputed)
    X_test = scaler.transform(X_test_imputed)
    
    # --- 3. THE EXPERIMENT GRIDS WITH PIPELINES (Prevents Leakage) ---
    models = {
        "Tuned Logistic Regression (Optuna)": (
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))
            ]),
            {
                'clf__C': [0.088987],
                'clf__penalty': ['l2']
            }
        ),
        "Tuned MLP (Optuna)": (
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', MLPClassifier(max_iter=1000, random_state=42))
            ]), 
            {
                'clf__hidden_layer_sizes': [(16,)],
                'clf__learning_rate_init': [0.000287],
                'clf__alpha': [0.000231]
            }
        ),
        "Tuned XGBoost (Optuna)": (
            ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(eval_metric='logloss', random_state=42))
            ]), 
            {
                'clf__n_estimators': [121],
                'clf__max_depth': [4],
                'clf__learning_rate': [0.005335],
                'clf__subsample': [0.666782],
                'clf__colsample_bytree': [0.662423],
                'clf__min_child_weight': [1]
            }
        )
    }
    
    # --- 4. AUTOMATED TRAINING & TUNING ---
    results = []
    print("\nStarting Dynamic Grid Search (Auto-expanding edges with Math Guardrails)...")
    for name, (model, param_grid) in models.items():
        print(f"Tuning {name}...")
        
        # We pass the full Train+Val set; GridSearchCV handles internal splitting naturally
        gs = dynamic_grid_search(model, param_grid, X_train_val, y_train_val, cv=5, max_expansions=3)
        best_model = gs.best_estimator_
        
        # Evaluate strictly on the untouched Test set
        metrics = evaluate_model(name, best_model, X_test, y_test)
        results.append(metrics)
        print(f"Finished! Ultimate settings found: {gs.best_params_}")
        
    # --- 5. DISPLAY RESULTS ---
    print("\n" + "="*80)
    print("PHASE 1 RESULTS: DYNAMICALLY TUNED CLINICAL BASELINES")
    print("(Evaluated strictly on the untouched Test Set)")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()