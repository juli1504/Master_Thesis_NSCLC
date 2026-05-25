"""
Phase 1a: Global Baselines (Enriched Clinical Metadata)
This script evaluates baseline performance and generates feature importance plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

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

def evaluate_model(name, model, X_test, y_test, X_train, y_train):
    """
    Advanced metrics suite: Binary support, Laplacian Baseline comparison,
    Calibration verification, and Operating Point analysis.
    """
    # 1. Predictions
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # 2. Extract probability of positive class (Binary)
    # y_probs is shape (n_samples, 2), we need column 1
    y_probs_pos = y_probs[:, 1]
    
    # 3. Metrics
    f1_val = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_probs_pos)
    
    # 4. Laplacian Baseline (Smoothing alpha=1)
    # Calculate frequency of class 1 in the balanced training set
    prior_pos = np.mean(y_train == 1)
    laplacian_probs = np.full(len(y_test), prior_pos)
    auc_baseline = roc_auc_score(y_test, laplacian_probs)
    
    # 5. Calibration Error (Mean squared difference)
    prob_true, prob_pred = calibration_curve(y_test, y_probs_pos, n_bins=5)
    # Handle cases where some bins might be empty
    bin_diffs = (prob_true - prob_pred)**2
    calibration_score = np.mean(bin_diffs[~np.isnan(bin_diffs)])
    
    return {
        "Model": name,
        "F1": f"{f1_val * 100:.1f}%",
        "AUC": f"{auc_val:.3f}",
        "Baseline AUC": f"{auc_baseline:.3f}",
        "Calibration Error": f"{calibration_score:.4f}"
    }

def main():
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    # Robust target detection
    possible_names = ['histology ', 'Histology ', 'histology', 'Histology']
    target_col = next((col for col in possible_names if col in df.columns), None)
    df['target'] = LabelEncoder().fit_transform(df[target_col])
    
    # Features
    num_cols = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_cols = ['Gender', 'Ethnicity', 'Smoking status']
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    X_raw = df[num_cols + cat_cols].copy()
    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    
    train_mask = df['dataset_split'] == 'Train'
    test_mask = df['dataset_split'] == 'Test'
    X_train_raw, y_train = X_encoded[train_mask], df.loc[train_mask, 'target']
    X_test_raw, y_test = X_encoded[test_mask], df.loc[test_mask, 'target']
    
    # Pipeline
    knn_imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(knn_imputer.fit_transform(X_train_raw))
    X_test = scaler.transform(knn_imputer.transform(X_test_raw))
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "Simple MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    smote = SMOTE(random_state=42)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)
    
    results = []
    # Inside main(), where you call the function:
    for name, model in models.items():
        model.fit(X_train_b, y_train_b)
        # Pass the required training data for the Laplacian baseline calculation
        results.append(evaluate_model(name, model, X_test, y_test, X_train_b, y_train_b))
        if name == "XGBoost":
            plot_feature_importance(model, X_encoded.columns, "feature_importance_1a")
        
    print("\n" + "="*70 + "\nPHASE 1a RESULTS: ENRICHED CLINICAL DATA\n" + "="*70)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()