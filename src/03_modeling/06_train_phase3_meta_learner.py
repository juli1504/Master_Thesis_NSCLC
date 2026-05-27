import os
import random
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import torchvision.models as models
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Define the paths explicitly here!
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
FILE_CLINICAL = PROJECT_ROOT / "data" / "raw" / "clinical" / "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
VISION_WEIGHTS = PROJECT_ROOT / "best_resnet_unfrozen_3.pth"
CLINICAL_MODEL = "best_clinical_model.pkl"

# --- 2. HELPERS (set_seed, build_resnet, evaluate_fusion, CTPatchDataset) ---
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

# --- 3. DATASET CLASS ---
class CTPatchDataset(Dataset):
    def __init__(self, manifest_df, label_encoder, clinical_data):
        # Filter for rows that actually have patches
        self.df = manifest_df[manifest_df['patch_extracted'] == True].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.le = label_encoder
        self.clinical = clinical_data
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = PROJECT_ROOT / row['patch_file_path']
        patch_array = np.load(patch_path).astype(np.float32)
        
        image_tensor = torch.tensor(patch_array)
        # Ensure channel-first format (C, H, W)
        # Ensure this part is NOT modifying your channels if it doesn't need to:
        if image_tensor.shape[-1] < 10:
            image_tensor = image_tensor.permute(2, 0, 1)
            
        label = self.le.transform([row['histology']])[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Return clinical data for this specific index
        return image_tensor, torch.tensor(self.clinical[idx], dtype=torch.float32), label_tensor

def build_resnet(in_channels, num_classes=2):
    model = models.resnet18()
    # Explicitly set the first layer to 7 channels to match your checkpoint
    model.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_fusion(y_true, y_probs, y_test_mask_len):
    # Ensure all imports needed for this (resample, roc_curve, etc.) are at the top!
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, roc_auc_score, f1_score
    from sklearn.calibration import calibration_curve
    from sklearn.utils import resample
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    acc, auc, f1 = accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_probs), f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn) if (tp + fn) > 0 else 0.0, tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    idx_90 = np.argmin(np.abs(tpr - 0.90))
    fpr_at_90 = fpr[idx_90]
    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=5)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    boot_f1s = [f1_score(*resample(y_true, (y_probs >= best_thresh).astype(int))) for _ in range(100)]
    ci_l, ci_h = np.percentile(boot_f1s, [2.5, 97.5])
    
    majority_class = 1 if np.sum(y_true) > (len(y_true) / 2) else 0
    laplacian_auc = roc_auc_score(y_true, np.full_like(y_true, majority_class))
    
    return acc, auc, f1, sens, spec, fpr_at_90, ece, ci_l, ci_h, laplacian_auc

class MultimodalFusionNet(nn.Module):
    def __init__(self, vision_model, clinical_model_path, num_classes=2):
        super().__init__()
        # 1. Vision Pillar (Frozen)
        self.vision_encoder = nn.Sequential(*list(vision_model.children())[:-1])
        for p in self.vision_encoder.parameters(): p.requires_grad = False
        
        # 2. Clinical Pillar (Loaded from your champion)
        champion = joblib.load(clinical_model_path)
        mlp = champion.named_steps['clf']
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(mlp.coefs_[0].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Inject learned clinical weights
        with torch.no_grad():
            self.clinical_encoder[0].weight.copy_(torch.from_numpy(mlp.coefs_[0].T))
            self.clinical_encoder[0].bias.copy_(torch.from_numpy(mlp.intercepts_[0]))
            self.clinical_encoder[2].weight.copy_(torch.from_numpy(mlp.coefs_[1].T))
            self.clinical_encoder[2].bias.copy_(torch.from_numpy(mlp.intercepts_[1]))
        for p in self.clinical_encoder.parameters(): p.requires_grad = False
        
        # 3. Fusion Head (The part that learns the weights!)
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, clinical_data):
        v_feat = self.vision_encoder(images).view(images.size(0), -1)
        c_feat = self.clinical_encoder(clinical_data)
        return self.fusion_head(torch.cat((v_feat, c_feat), dim=1))

# --- 4. MAIN EXECUTION ---
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    manifest_df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    clinical_df = pd.read_csv(FILE_CLINICAL)
    df = pd.merge(manifest_df[manifest_df['histology'].isin(['Adenocarcinoma', 'Squamous cell carcinoma'])], 
                  clinical_df, left_on='subject_id', right_on='Case ID', how='inner')
    
    # Define features (MUST MATCH Phase 1b)
    num_features = ['Age at Histological Diagnosis', 'Weight (lbs)', 'Pack Years', 'Quit Smoking Year']
    cat_features = ['Gender', 'Ethnicity', 'Smoking status']
    for col in num_features:
        df[col] = pd.to_numeric(df[col].replace(['Not Collected', 'Unknown', ' '], np.nan), errors='coerce')

    # 2. Extract Preprocessor from Champion Clinical Model
    champion = joblib.load(CLINICAL_MODEL)
    preprocessor = champion.named_steps['prep'] # Ensure this key matches your pipeline
    
    # Transform clinical data using the exact fitted preprocessor
    X_processed = preprocessor.transform(df[num_features + cat_features])
    
    # 3. Setup Encoders and Data
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['histology'])
    
    train_mask = df['dataset_split'].isin(['Train', 'Validation'])
    test_mask = df['dataset_split'] == 'Test'
    
    # Use the processed X and labels
    train_ds = CTPatchDataset(df[train_mask], le, X_processed[train_mask])
    test_ds = CTPatchDataset(df[test_mask], le, X_processed[test_mask])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 4. Initialize Fusion
    vision_model = build_resnet(in_channels=7).to(device)
    vision_model.load_state_dict(torch.load(VISION_WEIGHTS, map_location=device))
    
    # Initialize Net (Pass the champion path)
    fusion_net = MultimodalFusionNet(vision_model, CLINICAL_MODEL).to(device)
    
    # 5. Training Loop
    optimizer = optim.Adam(fusion_net.fusion_head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Deep Fusion Head...")
    for epoch in range(15):
        fusion_net.train()
        for img, clin, label in train_loader:
            optimizer.zero_grad()
            out = fusion_net(img.to(device), clin.to(device))
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            
    # 6. Evaluation
    fusion_net.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for img, clin, label in test_loader:
            out = fusion_net(img.to(device), clin.to(device))
            all_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(label.numpy())
            
    # Evaluate using your helper (ensure it is defined in the script)
    acc, auc, f1, sens, spec, fpr90, ece, ci_l, ci_h, laplacian_auc = evaluate_fusion(
        np.array(all_labels), np.array(all_probs), len(all_labels)
    )
    
    print(f"\nFusion Results: AUC: {auc:.3f}, Acc: {acc*100:.1f}%, Sens: {sens*100:.1f}%")
    
    # 6. Report
    # 6. Report Results for Thesis
    print("\n" + "="*85)
    print("PHASE 3c: MULTIMODAL DEEP FEATURE FUSION RESULTS")
    print("="*85)
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 40)
    print(f"{'AUC':<25} | {auc:.3f}")
    print(f"{'Accuracy':<25} | {acc*100:.1f}%")
    print(f"{'Sensitivity':<25} | {sens*100:.1f}%")
    print(f"{'Specificity':<25} | {spec*100:.1f}%")
    print(f"{'F1-Score':<25} | {f1*100:.1f}%")
    print(f"{'FPR @ 90% Sensitivity':<25} | {fpr90:.3f}")
    print(f"{'Calibration Error (ECE)':<25} | {ece:.3f}")
    print(f"{'F1-Score 95% CI':<25} | [{ci_l:.2f}, {ci_h:.2f}]")
    print("="*85)

if __name__ == "__main__":
    main()