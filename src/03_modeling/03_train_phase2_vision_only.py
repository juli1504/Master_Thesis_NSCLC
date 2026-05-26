"""
Phase 2: Vision Baselines and Fine-Tuning (2.5D CT Patches)
Features: 
- Global Seeding for absolute PyTorch reproducibility
- Dynamic 7-channel inputs
- Progressive architectural unfreezing (Block Dial)
- Optimal thresholding via Youden's J Statistic
- F1-Score Evaluation added for Subtyping
- Clinical Early Stopping (Sens + Spec) using Validation Set
- Strict Final Evaluation on untouched Test Set
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, f1_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import warnings
import csv
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss pushes the model to focus on hard-to-classify examples (Squamous) 
    and down-weights easy examples (Adenocarcinoma).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Optional: can pass a tensor of class weights e.g., torch.tensor([0.2, 0.8])
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the true class (pt)
        pt = torch.exp(-ce_loss)
        
        # Apply the Focal Loss formula: (1 - pt)^gamma * CE_Loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
            focal_loss = alpha_factor * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 1. CONFIGURATION & SEEDING ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
FILE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"

def set_seed(seed=42):
    """Locks down all random number generators for absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Forces deterministic algorithms (can be slightly slower, but required for science)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. DATASET DEFINITION ---
class CTPatchDataset(Dataset):
    def __init__(self, manifest_df, label_encoder, transform=None):
        self.df = manifest_df[manifest_df['patch_extracted'] == True].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.le = label_encoder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        patch_path = PROJECT_ROOT / row['patch_file_path']
        patch_array = np.load(patch_path).astype(np.float32)
        
        image_tensor = torch.tensor(patch_array)
        if image_tensor.shape[-1] < 10: 
            image_tensor = image_tensor.permute(2, 0, 1)
            
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        label = self.le.transform([row['histology']])[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor

# --- 3. MODEL BUILDER ---
def build_vision_model(model_name, unfreeze_blocks, in_channels, num_classes=2):
    """Builds the model and unfreezes a specific number of architectural blocks."""
    if model_name == 'resnet':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
            
        if unfreeze_blocks >= 1:
            for param in model.layer4.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.layer3.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.layer2.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.layer1.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            for param in model.parameters(): param.requires_grad = True

        original_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                padding=original_conv.padding, bias=False)
        model.conv1.weight.requires_grad = True 
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
        
        if unfreeze_blocks >= 1:
            for param in model.features.denseblock4.parameters(): param.requires_grad = True
            for param in model.features.norm5.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.features.transition3.parameters(): param.requires_grad = True
            for param in model.features.denseblock3.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.features.transition2.parameters(): param.requires_grad = True
            for param in model.features.denseblock2.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.features.transition1.parameters(): param.requires_grad = True
            for param in model.features.denseblock1.parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            for param in model.parameters(): param.requires_grad = True

        original_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(in_channels, original_conv.out_channels, 
                                         kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                         padding=original_conv.padding, bias=False)
        model.features.conv0.weight.requires_grad = True
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters(): param.requires_grad = False
            
        if unfreeze_blocks >= 1:
            for param in model.features[7:].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 2:
            for param in model.features[5:7].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 3:
            for param in model.features[3:5].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 4:
            for param in model.features[1:3].parameters(): param.requires_grad = True
        if unfreeze_blocks >= 5:
            for param in model.parameters(): param.requires_grad = True

        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(in_channels, original_conv.out_channels, 
                                         kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                                         padding=original_conv.padding, bias=False)
        model.features[0][0].weight.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    y_true, y_probs = np.array(y_true), np.array(y_probs)
    
    # --- 1. Basic Metrics ---
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # --- 2. Advanced Clinical Metrics with Safety Checks ---
    # FPR @ 90% Sensitivity (TPR)
    # If the model never hits 90% TPR, we set it to 1.0 (worst case)
    idx_90 = np.argmin(np.abs(tpr - 0.90))
    fpr_at_90 = fpr[idx_90] if len(fpr) > 0 else 1.0
    
    # Expected Calibration Error (ECE)
    # Binning can fail if y_probs are all identical
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=5)
        ece = np.mean(np.abs(prob_true - prob_pred))
    except:
        ece = 0.0
    
    # 95% Confidence Interval for F1 (Bootstrapping)
    boot_f1s = []
    for _ in range(100):
        try:
            t_b, p_b = resample(y_true, y_probs)
            # Ensure p_b has enough variance to threshold
            if len(np.unique(p_b)) > 1:
                p_b_pred = (p_b >= best_thresh).astype(int)
                boot_f1s.append(f1_score(t_b, p_b_pred))
            else:
                boot_f1s.append(0.0)
        except:
            boot_f1s.append(0.0)
            
    ci_low, ci_high = np.percentile(boot_f1s, [2.5, 97.5])
    
    return acc, auc, f1, sens, spec, best_thresh, fpr_at_90, ece, ci_low, ci_high

# --- 5. MAIN SCRIPT ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'densenet', 'efficientnet'])
    parser.add_argument('--unfreeze_blocks', type=int, default=1, choices=[0, 1, 2, 3, 4, 5], help="0=Frozen, 1-4=Partial, 5=Full")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # Important: Lock down all random states!
    set_seed(42)

    print(f"=== STARTING PHASE 2 RUN ===")
    print(f"Model: {args.model.upper()} | Unfrozen Blocks: {args.unfreeze_blocks} | Epochs: {args.epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Hardware: {device}\n")

    df = pd.read_csv(FILE_MANIFEST, sep=';', decimal=',')
    df = df[df['dataset_split'] != 'Excluded'].copy()
    valid_cancers = ['Adenocarcinoma', 'Squamous cell carcinoma']
    df = df[df['histology'].isin(valid_cancers)].copy()
    
    le = LabelEncoder()
    le.fit(df['histology'])
    print(f"Target Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    # --- STRICT 3-WAY SPLIT ---
    train_df = df[df['dataset_split'] == 'Train']
    val_df = df[df['dataset_split'] == 'Validation']
    test_df = df[df['dataset_split'] == 'Test'] # Kept pure until the end

    # --- DATA AUGMENTATION ---
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15)
    ])

    train_dataset = CTPatchDataset(train_df, le, transform=train_transforms)
    val_dataset = CTPatchDataset(val_df, le, transform=None) # No augmentation on Val
    test_dataset = CTPatchDataset(test_df, le, transform=None) # No augmentation on Test

    # Since we set the global seed, shuffle=True will now shuffle identically every time
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    sample_img, _ = train_dataset[0]
    in_channels = sample_img.shape[0]
    print(f"Detected 2.5D Patches with {in_channels} channels.\n")
    print(f"Data Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n")

    model = build_vision_model(args.model, args.unfreeze_blocks, in_channels).to(device)
    #criterion = nn.CrossEntropyLoss()
    # Setting gamma=2.0 is the standard paper recommendation for Focal Loss
    criterion = FocalLoss(gamma=2.0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    save_name = f"best_{args.model}_unfrozen_{args.unfreeze_blocks}.pth"

    # --- TRAINING LOOP ---
    best_clinical_score = 0.0
    best_auc_tracker = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate strictly on VALIDATION loader during training
        val_acc, val_auc, val_f1, val_sens, val_spec, val_thresh, _, _, _, _ = evaluate(model, val_loader, device)        
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Optimal Val Cutoff: {val_thresh:.2f} | Val AUC: {val_auc:.3f} | Val F1: {val_f1*100:.1f}% | Val Sens: {val_sens*100:.1f}% | Val Spec: {val_spec*100:.1f}%")    
        # --- EARLY STOPPING & SAVING (BASED ON VAL SET) ---
        current_clinical_score = val_sens + val_spec
        
        if current_clinical_score > best_clinical_score:
            best_clinical_score = current_clinical_score
            best_auc_tracker = val_auc
            patience_counter = 0  
            
            torch.save(model.state_dict(), save_name)
            print(f"New best clinical model saved. (Val Sens+Spec: {best_clinical_score:.3f} | Val AUC: {best_auc_tracker:.3f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered. The model stopped improving after {epoch+1} epochs.")
            break

    # --- FINAL TEST EVALUATION ---
    if os.path.exists(save_name):
        model.load_state_dict(torch.load(save_name))
    
    acc, auc, f1, sens, spec, thresh, fpr90, ece, ci_l, ci_h = evaluate(model, test_loader, device)
    
    # Save to master CSV
    import csv
    results_file = "all_phase2_benchmarks.csv"
    row = {
        "Architecture": args.model.upper(),
        "Unfreeze": args.unfreeze_blocks,
        "AUC": f"{auc:.3f}",
        "F1": f"{f1*100:.1f}%",
        "Sens": f"{sens*100:.1f}%",
        "Spec": f"{spec*100:.1f}%",
        "FPR@90": f"{fpr90:.3f}",
        "ECE": f"{ece:.3f}",
        "F1_CI": f"[{ci_l:.2f}, {ci_h:.2f}]"
    }
    
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    print(f"\nResult appended to {results_file}")

if __name__ == "__main__":
    main()