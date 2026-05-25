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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
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

def evaluate_model(name, model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    return {"Model": name, "Accuracy": f"{acc * 100:.1f}%", "AUC": f"{auc:.3f}", 
            "F1": f"{f1 * 100:.1f}%", "Sensitivity": f"{sensitivity * 100:.1f}%", 
            "Specificity": f"{specificity * 100:.1f}%"}

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
    for name, model in models.items():
        model.fit(X_train_b, y_train_b)
        results.append(evaluate_model(name, model, X_test, y_test))
        if name == "XGBoost":
            plot_feature_importance(model, X_encoded.columns, "feature_importance_1a")
        
    print("\n" + "="*70 + "\nPHASE 1a RESULTS: ENRICHED CLINICAL DATA\n" + "="*70)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()