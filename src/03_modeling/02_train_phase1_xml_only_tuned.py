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
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import joblib
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

def main():
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    df = df[df['dataset_split'] != 'Excluded'].copy()
    
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    # Update the OrdinalEncoder inside your preprocessor:
    preprocessor = ColumnTransformer(transformers=[
    ('num', SklearnPipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), num_features),
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
    
    models = {
    "Tuned LR": (ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(random_state=42))]), {'clf__C': [0.1]}),
    "Tuned XGBoost": (
        ImbPipeline([
            ('prep', preprocessor), 
            ('smote', SMOTE(random_state=42)), 
            ('clf', XGBClassifier(eval_metric='logloss', random_state=42))
        ]), 
        {
            'clf__n_estimators': [100],
            'clf__reg_alpha': [0.1, 1.0, 10.0],
            'clf__reg_lambda': [0.1, 1.0, 10.0]
        }
    ),
    "TabPFN": (SklearnPipeline([('prep', preprocessor), ('clf', TabPFNClassifier(device='cpu', model_path="03_modeling/tabpfn-v2.5-classifier-v2.5_default.ckpt"))]), {}),
    # Wrap this in a tuple so it matches the (model, grid) structure
    "Tuned MLP": (
        ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=0.01, alpha=0.01, max_iter=1000, random_state=42))]), 
        {}
    )}
    
    # 1. Generate the balanced training set
    smote = SMOTE(random_state=42)
    # We transform the data first to match the pipeline input
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_train_b, y_train_b = smote.fit_resample(X_train_transformed, y_train)

    # 2. Execution Loop
    results = []
    best_auc = 0.0
    best_model_overall = None
    
    for name, (model, grid) in models.items():
        if name == "TabPFN":
            model.fit(X_train, y_train)
            best_model = model
        else:
            gs = GridSearchCV(model, grid, cv=3, scoring='roc_auc', n_jobs=1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            
        # Evaluation
        res = evaluate_model_advanced(name, best_model, X_test, y_test, X_train_b, y_train_b)
        
        # Track best model based on AUC (converting string "0.XXX" to float)
        current_auc = float(res["AUC"])
        if current_auc > best_auc:
            best_auc = current_auc
            best_model_overall = best_model
            print(f"New best model found: {name} (AUC: {best_auc:.3f})")
        
        # Bootstrap CI (using single-threaded resample)
        preds = best_model.predict(X_test)
        scores = [f1_score(*resample(y_test, preds)) for _ in range(500)]
        res["F1 95% CI"] = f"[{np.percentile(scores, 2.5):.2f}, {np.percentile(scores, 97.5):.2f}]"
        results.append(res)
        
        if name == "Tuned XGBoost": 
            plot_feature_importance(best_model, num_features, cat_features, "feature_importance_1b")

    # 3. Save the best model after the loop finishes
    joblib.dump(best_model_overall, "best_clinical_model.pkl")
    print(f"\nFinal best model saved: best_clinical_model.pkl (AUC: {best_auc:.3f})")

    print(f"\n{'='*70}")
    print("PHASE 1b: FINAL CLINICAL BASELINE RESULTS (UNIFIED PREPROCESSING)")
    print(f"{'='*70}")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()