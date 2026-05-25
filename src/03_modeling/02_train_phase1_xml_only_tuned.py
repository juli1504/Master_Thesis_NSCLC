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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.calibration import calibration_curve
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

def evaluate_model_advanced(name, model, X_test, y_test, X_train, y_train):
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
    
    # 1. Generate the balanced training set
    smote = SMOTE(random_state=42)
    # We transform the data first to match the pipeline input
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_train_b, y_train_b = smote.fit_resample(X_train_transformed, y_train)

    # 2. Execution Loop
    results = []
    for name, (model, param_grid) in models.items():
        if name == "TabPFN":
            # TabPFN handles its own prep, so we pass raw X_train
            model.fit(X_train, y_train)
            results.append(evaluate_model_advanced(name, model, X_test, y_test, X_train, y_train))
        else:
            # GridSearchCV models are wrapped in pipelines that include 'prep'
            gs = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            
            # Pass X_train_b/y_train_b here to ensure the baseline 
            # reflects the distribution the model actually saw
            results.append(evaluate_model_advanced(name, best_model, X_test, y_test, X_train_b, y_train_b))
            
            if name == "Tuned XGBoost":
                plot_feature_importance(best_model, num_features + cat_features, "feature_importance_1b")

    print("\nPHASE 1b: FINAL CLINICAL BASELINE RESULTS (UNIFIED PREPROCESSING)")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()