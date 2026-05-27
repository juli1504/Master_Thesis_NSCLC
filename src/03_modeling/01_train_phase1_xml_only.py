"""
Phase 1a: Global Baselines (Enriched Clinical Metadata)
This script evaluates baseline performance and generates feature importance plots.
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
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def plot_feature_importance(model, feature_names, filename):
    """Saves a bar chart of XGBoost feature importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title(f"XGBoost Feature Importance")
    plt.xlabel("Importance Score")
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

def main():
    # Load and Prepare Data
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    df = df[df['dataset_split'] != 'Excluded'].copy()
    
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
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
    
    train_mask = df['dataset_split'] == 'Train'
    test_mask = df['dataset_split'] == 'Test'
    X_train, y_train, X_test, y_test = X[train_mask], y[train_mask], X[test_mask], y[test_mask]
    
    # Precompute Balanced training set for Laplacian Baseline
    X_train_transformed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_b, y_train_b = smote.fit_resample(X_train_transformed, y_train)

    models = {
        "Logistic Regression": (ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(random_state=42))]), {}),
        "Simple MLP": (ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42))]), {}),
        "XGBoost": (ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', XGBClassifier(eval_metric='logloss', random_state=42))]), {})
    }
    
    results = []
    print(f"\n{'='*70}\nPHASE 1a: ADVANCED CLINICAL EVALUATION\n{'='*70}")
    
    for name, (model, param_grid) in models.items():
        model.fit(X_train, y_train)
        
        # 1. Advanced Metrics
        res = evaluate_model_advanced(name, model, X_test, y_test, X_train_b, y_train_b)
        
        # 2. Bootstrap Confidence Intervals
        y_pred = model.predict(X_test)
        boot_scores = []
        for _ in range(1000):
            y_t_res, y_p_res = resample(y_test, y_pred)
            boot_scores.append(f1_score(y_t_res, y_p_res))
        ci = np.percentile(boot_scores, [2.5, 97.5])
        
        res["F1 95% CI"] = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        results.append(res)
            
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()