import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
import atexit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 320, 240
BATCH_SIZE = 32
EPOCHS = 15
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data Loaders
def get_loaders():
    train_data = datasets.ImageFolder(
        '../data/raw/dataset2-master/TRAIN',
        transform=train_transform
    )
    test_data = datasets.ImageFolder(
        '../data/raw/dataset2-master/TEST',
        transform=test_transform
    )
    
    with open('../models/class_mappings.json', 'w') as f:
        json.dump(train_data.class_to_idx, f)
    
    return (
        DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                  num_workers=0, pin_memory=True),  # num_workers=0 for Windows
        DataLoader(test_data, batch_size=BATCH_SIZE, 
                 num_workers=0, pin_memory=True)
    )

# Model
class WBCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.base = models.resnet18(weights=weights)
        
        # Modify first layer for 320x240 input
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, len(CLASS_NAMES))
        
        # Freeze early layers
        for param in list(self.base.parameters())[:-4]:
            param.requires_grad = False

    def forward(self, x):
        return self.base(x)

# Training
def train():
    # Initialize
    torch.manual_seed(42)
    train_loader, test_loader = get_loaders()
    model = WBCClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Cleanup handler
    def cleanup():
        if hasattr(train_loader, '_iterator'):
            train_loader._iterator._shutdown_workers()
        if hasattr(test_loader, '_iterator'):
            test_loader._iterator._shutdown_workers()
    atexit.register(cleanup)
    
    try:
        # Training loop
        best_accuracy = 0
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Print batch progress
                print(f'Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}', end='\r')
            
            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images.to(DEVICE))
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels)
            
            # Calculate metrics
            val_preds = torch.cat(all_preds)
            val_labels = torch.cat(all_labels)
            val_accuracy = (torch.argmax(val_preds, 1) == val_labels).float().mean()
            scheduler.step(train_loss/len(train_loader))
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), '../models/wbc_classifier.pth')
            
            # Epoch summary
            print(f'\nEpoch {epoch+1}/{EPOCHS} | '
                  f'Train Loss: {train_loss/len(train_loader):.4f} | '
                  f'Val Acc: {val_accuracy:.4f} | '
                  f'Time: {(time.time()-epoch_start)/60:.1f}m')
    
    except KeyboardInterrupt:
        print("\nTraining stopped manually. Saving model...")
        torch.save(model.state_dict(), '../models/wbc_classifier_interrupt.pth')
    
    # Final evaluation
    evaluate(model, test_loader)

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            all_preds.append(outputs.cpu())
            all_labels.append(labels)
    
    # Generate metrics
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.argmax(torch.cat(all_preds), 1).numpy()
    y_probs = torch.softmax(torch.cat(all_preds), 1).numpy()
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.savefig('../evaluation/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Save metrics
    os.makedirs('../evaluation', exist_ok=True)
    with open('../evaluation/metrics.json', 'w') as f:
        json.dump({
            'accuracy': float((y_true == y_pred).mean()),
            'roc_auc': float(roc_auc_score(y_true, y_probs, multi_class='ovo')),
            'report': classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        }, f, indent=2)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Windows-safe
    os.makedirs('../models', exist_ok=True)
    train()