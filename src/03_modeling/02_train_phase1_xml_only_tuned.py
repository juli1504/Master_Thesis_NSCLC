"""
Phase 1b: Final Clinical Baseline (Enriched Clinical Metadata)
This script evaluates baseline performance, tunes hyperparameters, 
and generates feature importance plots for the XGBoost model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import resample
import joblib
import optuna
# This forces everything that uses joblib to run on a single core
joblib.parallel_backend('sequential')

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def plot_feature_importance(model, num_features, cat_features, filename):
    """Saves a bar chart using the actual feature names from the pipeline."""
    prep = model.named_steps['prep']
    # Get OHE feature names using the passed cat_features list
    cat_names = prep.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)
    
    # Now num_features is passed in as an argument
    all_feature_names = num_features + list(cat_names)
    
    clf = model.named_steps['clf']
    importances = pd.Series(clf.feature_importances_, index=all_feature_names)
    
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).plot(kind='barh', color='salmon')
    plt.title(f"XGBoost Feature Importance (Tuned Phase 1b)")
    plt.tight_layout()
    save_path = PROJECT_ROOT / "results" / "figures" / f"{filename}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Feature importance saved to {save_path}")

def evaluate_model_advanced(name, model, X_test, y_test, X_train, y_train):
    """
    Advanced metrics suite including operating point analysis and Laplacian baseline.
    """
    # 1. Predictions
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    y_probs_pos = y_probs[:, 1]
    
    # 2. Performance Metrics
    f1_val = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_probs_pos)
    
    # 3. FPR at Fixed TPR (Operating Point: 90% Sensitivity)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs_pos)
    target_tpr = 0.90
    # Find index closest to 0.90 TPR
    idx = np.argmin(np.abs(tpr - target_tpr))
    fixed_fpr = fpr[idx]
    
    # 4. Laplacian Baseline (Prior probability from training set)
    prior_pos = np.mean(y_train == 1)
    laplacian_probs = np.full(len(y_test), prior_pos)
    auc_baseline = roc_auc_score(y_test, laplacian_probs)
    
    # 5. Calibration Error
    prob_true, prob_pred = calibration_curve(y_test, y_probs_pos, n_bins=5)
    bin_diffs = (prob_true - prob_pred)**2
    calibration_score = np.mean(bin_diffs[~np.isnan(bin_diffs)])
    
    return {
        "Model": name,
        "F1": f"{f1_val * 100:.1f}%",
        "AUC": f"{auc_val:.3f}",
        "FPR@90%TPR": f"{fixed_fpr:.2f}",
        "Baseline AUC": f"{auc_baseline:.3f}",
        "Calib. Error": f"{calibration_score:.4f}"
    }

def objective(trial, model_name, pipeline, X_train, y_train):
    """Optuna objective: defines dynamic search space and CV evaluation."""
    if model_name == "Tuned XGBoost":
        params = {
            'clf__n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'clf__max_depth': trial.suggest_int('max_depth', 3, 10),
            'clf__learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            'clf__subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
    elif model_name == "Tuned MLP":
        params = {
            'clf__hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 32), (32, 16), (128, 64)]),
            'clf__learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'clf__alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        }
    else: # Default/LR
        params = {'clf__C': trial.suggest_float('C', 0.01, 10.0, log=True)}
        
    pipeline.set_params(**params)
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()

def main():
    # 1. Data Loading and Cleaning
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    df = df[df['dataset_split'] != 'Excluded'].copy()
    
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    # 2. Pipeline Definition
    preprocessor = ColumnTransformer(transformers=[
        ('num', SklearnPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', SklearnPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_features)
    ])
    
    X = df[num_features + cat_features]
    target_col = 'histology ' if 'histology ' in df.columns else 'Histology '
    y = LabelEncoder().fit_transform(df[target_col])
    
    X_train, y_train = X[df['dataset_split'] == 'Train'], y[df['dataset_split'] == 'Train']
    X_test, y_test = X[df['dataset_split'] == 'Test'], y[df['dataset_split'] == 'Test']
    
    # 3. Model Configurations
    models_config = {
        "Tuned LR": ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(random_state=42))]),
        "Tuned XGBoost": ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', XGBClassifier(eval_metric='logloss', random_state=42))]),
        "Tuned MLP": ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', MLPClassifier(max_iter=1000, random_state=42))])
    }

    results = []
    best_auc = 0.0
    best_model_overall = None
    
    # 4. Optimization and Training Loop
    for name, pipeline in models_config.items():
        print(f"\n--- Optimizing {name} with Optuna ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, name, pipeline, X_train, y_train), n_trials=30)
        
        # Fit best model on the full training set
        best_pipeline = pipeline
        mapped_params = {f"clf__{k}": v for k, v in study.best_params.items()}
        best_pipeline.set_params(**mapped_params)
        best_pipeline.fit(X_train, y_train)
        
        # 5. Advanced Evaluation
        res = evaluate_model_advanced(name, best_pipeline, X_test, y_test, X_train, y_train)
        
        # Bootstrap CI
        preds = best_pipeline.predict(X_test)
        scores = [f1_score(*resample(y_test, preds)) for _ in range(500)]
        res["F1 95% CI"] = f"[{np.percentile(scores, 2.5):.2f}, {np.percentile(scores, 97.5):.2f}]"
        results.append(res)
        
        if float(res["AUC"]) > best_auc:
            best_auc, best_model_overall = float(res["AUC"]), best_pipeline
            
        if name == "Tuned XGBoost": 
            plot_feature_importance(best_pipeline, num_features, cat_features, "feature_importance_1b")

    # 3. Save the best model after the loop finishes
    joblib.dump(best_model_overall, "best_clinical_model.pkl")
    print(f"\nFinal best model saved: best_clinical_model.pkl (AUC: {best_auc:.3f})")

    print(f"\n{'='*70}")
    print("PHASE 1b: FINAL CLINICAL BASELINE RESULTS (UNIFIED PREPROCESSING)")
    print(f"{'='*70}")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()