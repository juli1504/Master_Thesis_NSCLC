"""
Phase 1b: Final Clinical Baseline (Enriched Clinical Metadata)
This script evaluates baseline performance, tunes hyperparameters, 
and generates feature importance plots for the XGBoost model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

def plot_feature_importance(model, feature_names, filename):
    """Saves a bar chart of XGBoost feature importance from a pipeline."""
    # Access the classifier step from the pipeline
    clf = model.named_steps['clf']
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    plt.figure(figsize=(10, 6))
    importances.nlargest(10).plot(kind='barh', color='salmon')
    plt.title(f"XGBoost Feature Importance (Tuned Phase 1b)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    save_path = PROJECT_ROOT / "results" / "figures" / f"{filename}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Feature importance saved to {save_path}")

def evaluate_model(name, model, X_test, y_test):
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_probs = model.predict(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    return {"Model": name, "Accuracy": f"{acc*100:.1f}%", "AUC": f"{auc:.3f}", 
            "F1": f"{f1*100:.1f}%", "Sensitivity": f"{sensitivity*100:.1f}%", 
            "Specificity": f"{specificity*100:.1f}%"}

def main():
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    df = df[df['dataset_split'] != 'Excluded'].copy()
    
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', SklearnPipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), num_features),
        ('cat', SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_features)
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
        ),        "TabPFN": (SklearnPipeline([('prep', preprocessor), ('clf', TabPFNClassifier(device='cpu', model_path="03_modeling/tabpfn-v2.5-classifier-v2.5_default.ckpt"))]), {})
    }
    
    results = []
    for name, (model, param_grid) in models.items():
        if name == "TabPFN":
            model.fit(X_train, y_train)
            results.append(evaluate_model(name, model, X_test, y_test))
        else:
            gs = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            results.append(evaluate_model(name, best_model, X_test, y_test))
            if name == "Tuned XGBoost":
                plot_feature_importance(best_model, num_features + cat_features, "feature_importance_1b")
            
    print("\nPHASE 1b: FINAL CLINICAL BASELINE RESULTS (UNIFIED PREPROCESSING)")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()