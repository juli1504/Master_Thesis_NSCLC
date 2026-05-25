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
import warnings
warnings.filterwarnings('ignore')

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

# --- 4. EVALUATION FUNCTION (Youden's J STATISTIC + F1) ---
def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5  
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    best_thresh = thresholds[optimal_idx]
    
    y_pred = (y_probs >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return acc, auc, f1, sens, spec, best_thresh

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
    criterion = nn.CrossEntropyLoss()
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
        val_acc, val_auc, val_f1, val_sens, val_spec, best_thresh = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Optimal Val Cutoff: {best_thresh:.2f} | Val AUC: {val_auc:.3f} | Val F1: {val_f1*100:.1f}% | Val Sens: {val_sens*100:.1f}% | Val Spec: {val_spec*100:.1f}%")

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
    print("\n" + "="*70)
    print("PHASE 2 FINAL RESULTS: EVALUATING ON UNTOUCHED TEST SET")
    print("="*70)
    
    # Load the absolute best weights identified by the Validation set
    model.load_state_dict(torch.load(save_name))
    
    # Evaluate exactly once on pure Test set
    test_acc, test_auc, test_f1, test_sens, test_spec, test_thresh = evaluate(model, test_loader, device)
    
    print(f"Model: {args.model.upper()} | Unfrozen Blocks: {args.unfreeze_blocks}")
    print(f"Optimal Test Cutoff: {test_thresh:.2f}")
    print(f"Test Accuracy:    {test_acc*100:.1f}%")
    print(f"Test F1-Score:    {test_f1*100:.1f}%")
    print(f"Test AUC:         {test_auc:.3f}")
    print(f"Test Sensitivity: {test_sens*100:.1f}%")
    print(f"Test Specificity: {test_spec*100:.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()