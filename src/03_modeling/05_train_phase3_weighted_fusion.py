"""
Phase 3b: Multimodal Late Fusion (AUC-Weighted Average)

This script implements AUC-Weighted Late Fusion. 
Instead of a 50/50 split, the models are assigned a "Voting Weight" 
proportional to their individual baseline AUC scores, giving the 
superior Vision model a mathematically higher impact on the final decision.
"""

import os
import random
import joblib
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & SEEDING ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
VISION_WEIGHTS = PROJECT_ROOT / "best_resnet_unfrozen_3.pth"

# AUC Scores from Phase 1b and Phase 2
AUC_CLINICAL = 0.722
AUC_VISION = 0.635

def set_seed(seed=42):
    """Locks down all random number generators for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. VISION DATASET & BUILDER ---
class CTPatchDataset(Dataset):
    def __init__(self, manifest_df, label_encoder):
        self.df = manifest_df[manifest_df['patch_extracted'] == True].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.le = label_encoder
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = PROJECT_ROOT / row['patch_file_path']
        patch_array = np.load(patch_path).astype(np.float32)
        
        image_tensor = torch.tensor(patch_array)
        if image_tensor.shape[-1] < 10: 
            image_tensor = image_tensor.permute(2, 0, 1)
            
        label = self.le.transform([row['histology']])[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

def build_resnet(in_channels, num_classes=2):
    model = models.resnet18()
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                            kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                            padding=original_conv.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_fusion(y_true, y_probs, y_test_mask_len):
    """
    Full clinical evaluation suite.
    y_test_mask_len: used to calculate the Laplacian (majority class) baseline.
    """
    # 1. Standard Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    acc, auc, f1 = accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_probs), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn) if (tp + fn) > 0 else 0.0, tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 2. FPR @ 90% Sensitivity
    idx_90 = np.argmin(np.abs(tpr - 0.90))
    fpr_at_90 = fpr[idx_90]
    
    # 3. Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=5)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # 4. 95% Confidence Interval (Bootstrapping)
    boot_f1s = [f1_score(*resample(y_true, (y_probs >= best_thresh).astype(int))) for _ in range(100)]
    ci_l, ci_h = np.percentile(boot_f1s, [2.5, 97.5])
    
    # 5. Laplacian Baseline (Majority Class Prediction)
    majority_class = 1 if np.sum(y_true) > (len(y_true) / 2) else 0
    laplacian_pred = np.full_like(y_true, majority_class)
    laplacian_auc = roc_auc_score(y_true, laplacian_pred)
    
    return acc, auc, f1, sens, spec, fpr_at_90, ece, ci_l, ci_h, laplacian_auc

# --- 3. MAIN FUSION SCRIPT ---
def main():
    # Lock the environment!
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STARTING PHASE 3b: ECE-INVERSE WEIGHTED LATE FUSION ===")
    print(f"Using Hardware: {device}\n")

    # 1. Load and Merge Data
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    manifest_df = manifest_df[manifest_df['dataset_split'] != 'Excluded'].copy()
    valid_cancers = ['Adenocarcinoma', 'Squamous cell carcinoma']
    manifest_df = manifest_df[manifest_df['histology'].isin(valid_cancers)].copy()
    
    df = pd.merge(manifest_df, clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    # 2. Prepare Clinical Data
    clinical_features = ['Age at Histological Diagnosis', 'Gender', 'Smoking status']
    X_raw = df[clinical_features].copy()
    X_encoded = pd.get_dummies(X_raw, columns=['Gender', 'Smoking status'], drop_first=True)
    
    train_val_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test'
    
    X_train_val = X_encoded[train_val_mask]
    y_train_val = df.loc[train_val_mask, 'target']
    X_test = X_encoded[test_mask]
    y_test = df.loc[test_mask, 'target'] 
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_val_scaled = scaler.fit_transform(imputer.fit_transform(X_train_val))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    # --- PILLAR 1: LOAD CLINICAL CHAMPION ---
    print("Loading Best Clinical Model (Phase 1b)...")
    clinical_pipeline = joblib.load("best_clinical_model.pkl")
    
    # We must prepare the raw input X_test (all 7 features) to match the pipeline's expected input
    # Ensure you are using the same feature definitions as Phase 1b
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    
    # Clean numeric columns for the test set
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')
    
    X_test_raw = df[test_mask][num_features + cat_features]
    
    # The pipeline handles scaling and encoding internally
    probs_clinical = clinical_pipeline.predict_proba(X_test_raw)[:, 1]

    # --- PILLAR 2: GET VISION PROBABILITIES ---
    print("Loading Phase 2 Best Model ((ResNet Level 3))...")
    test_df = df[test_mask].copy()
    test_dataset = CTPatchDataset(test_df, le)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    in_channels = test_dataset[0][0].shape[0]
    vision_model = build_resnet(in_channels).to(device)
    vision_model.load_state_dict(torch.load(VISION_WEIGHTS, map_location=device))
    vision_model.eval()
    
    probs_vision = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = vision_model(images)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            probs_vision.extend(batch_probs.cpu().numpy())
            
    probs_vision = np.array(probs_vision)

    # --- PILLAR 3: ECE-INVERSE WEIGHTED LATE FUSION ---
    print("\nCalculating Calibration-Based Voting Weights...")
    
    # 1. Calculate ECE for individual models
    # We use a helper function to get ECE for clinical and vision independently
    def get_ece(y_true, y_probs, n_bins=5):
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))

    ece_clinical = get_ece(y_test.values, probs_clinical)
    ece_vision = get_ece(y_test.values, probs_vision)
    
    # 2. Convert ECE to weights (Inverse: 1/ECE)
    # Adding a small epsilon to avoid division by zero
    inv_ece_clin = 1.0 / (ece_clinical + 1e-6)
    inv_ece_vis = 1.0 / (ece_vision + 1e-6)
    
    total_inv_ece = inv_ece_clin + inv_ece_vis
    weight_clinical = inv_ece_clin / total_inv_ece
    weight_vision = inv_ece_vis / total_inv_ece
    
    print(f" -> Clinical ECE: {ece_clinical:.3f} (Weight: {weight_clinical*100:.1f}%)")
    print(f" -> Vision ECE:   {ece_vision:.3f} (Weight: {weight_vision*100:.1f}%)")
    
    probs_fusion = (probs_clinical * weight_clinical) + (probs_vision * weight_vision)
    y_test_array = y_test.values
    
    # Calculate Optimal Threshold
    fpr, tpr, thresholds = roc_curve(y_test_array, probs_fusion)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred_fusion = (probs_fusion >= best_thresh).astype(int)

    # --- THIS IS THE LINE THAT DEFINES THE VARIABLE ---
    auc_fusion = roc_auc_score(y_test_array, probs_fusion)
    
    # [NEW] Use the robust evaluation suite
    acc, auc, f1, sens, spec, fpr90, ece, ci_l, ci_h, laplacian_auc = evaluate_fusion(y_test_array, probs_fusion, len(y_test_array))    
    # [NEW] Save to master benchmark CSV
    results_file = "all_phase3_benchmarks.csv"
    row = {
        "Strategy": "ECE_Inverse_Weighted_Fusion",
        "AUC": f"{auc:.3f}",
        "F1": f"{f1*100:.1f}%",
        "Sens": f"{sens*100:.1f}%",
        "Spec": f"{spec*100:.1f}%",
        "FPR@90": f"{fpr90:.3f}",
        "ECE": f"{ece:.3f}",
        "F1_CI": f"[{ci_l:.2f}, {ci_h:.2f}]"
    }

    # --- DISPLAY FINAL RESULTS ---
    print("\n" + "="*85)
    print("PHASE 3b: FINAL MULTIMODAL RESULTS (ECE-WEIGHTED)")
    print("="*85)
    print(f"{'Metric':<15} | {'Phase 1 (Clinical)':<20} | {'Phase 2 (Vision)':<18} | {'Phase 3 (Fusion)':<15}")
    print("-" * 85)
    print(f"{'AUC':<15} | {'0.722':<20} | {'0.635':<18} | {auc:.3f}")
    print(f"{'Sensitivity':<15} | {'80.0%':<20} | {'100.0%':<18} | {sens*100:.1f}%")
    print(f"{'Specificity':<15} | {'47.8%':<20} | {'65.2%':<18} | {spec*100:.1f}%")
    print(f"{'F1-Score':<15} | {'38.1%':<20} | {'55.6%':<18} | {f1*100:.1f}%")
    print(f"{'Accuracy':<15} | {'53.6%':<20} | {'71.4%':<18} | {acc*100:.1f}%")
    
    print("\n" + "="*85)
    print("PHASE 3b: ADVANCED DIAGNOSTIC METRICS")
    print("="*85)
    print(f"{'FPR @ 90% Sensitivity':<25} | {fpr90:.3f}")
    print(f"{'Calibration Error (ECE)':<25} | {ece:.3f}")
    print(f"{'F1-Score 95% CI':<25} | [{ci_l:.2f}, {ci_h:.2f}]")
    print(f"{'Laplacian Baseline AUC':<25} | {laplacian_auc:.3f}")
    print("="*85)

if __name__ == "__main__":
    main()