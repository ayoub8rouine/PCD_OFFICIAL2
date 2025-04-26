import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import os
from tqdm import tqdm

# === DETECT WHITE CELLS WITH YOLO ===
def detect_white_cells_yolo(model_path, image_path, output_csv_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    img = Image.open(image_path).convert("RGB")
    results = model(img, size=640)
    detections = results.pandas().xyxy[0]

    csv_data = []
    for idx, row in detections.iterrows():
        csv_data.append({
            "class": int(row['class']),
            "xmin": int(row['xmin']),
            "xmax": int(row['xmax']),
            "ymin": int(row['ymin']),
            "ymax": int(row['ymax']),
        })

    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved detections to {output_csv_path}")

# === CROP WHITE CELLS FROM CSV ===
def crop_white_cells_from_csv(image_path, csv_path):
    img = Image.open(image_path).convert("RGB")
    df = pd.read_csv(csv_path)
    
    white_cell_crops = []
    for idx, row in df.iterrows():
        if row['class'] == 1:
            xmin, xmax = int(row['xmin']), int(row['xmax'])
            ymin, ymax = int(row['ymin']), int(row['ymax'])
            crop = img.crop((xmin, ymin, xmax, ymax))
            white_cell_crops.append(crop)
    
    return img, white_cell_crops

# === PATCH CLASSIFIER MODEL ===
class PatchClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.orig_features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.orig_features = nn.Sequential(*list(self.orig_features.children())[:-1])

        self.patch_features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.patch_features = nn.Sequential(*list(self.patch_features.children())[:-1])

        self.attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, originals, patch_lists):
        device = next(self.parameters()).device
        batch_features = []

        orig_feats = self.orig_features(originals.to(device)).squeeze()

        for i, patches in enumerate(patch_lists):
            patches = patches.to(device)
            feats = self.patch_features(patches).squeeze()
            if feats.ndim == 1:
                feats = feats.unsqueeze(0)

            weights = self.attention(feats)
            weighted_feat = (feats * weights).sum(dim=0)

            combined = torch.cat([orig_feats[i], weighted_feat], dim=0)
            batch_features.append(combined)

        return self.classifier(torch.stack(batch_features))

# === PREDICT PATCH CLASSIFIER ===
def predict_patch_classifier(model_path, original_image, patches, num_classes=4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PatchClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        orig_tensor = transform(original_image).unsqueeze(0).to(device)
        patch_tensors = [transform(patch) for patch in patches]
        patch_tensors = torch.stack(patch_tensors).unsqueeze(0)  # batch=1

        outputs = model(orig_tensor, [patch_tensors])
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.cpu().numpy()

# === FULL PIPELINE ===
def full_pipeline(yolo_model_path, patch_classifier_model_path, input_image_path):
    temp_csv = "temp_detections.csv"

    # Step 1: Detect white cells
    detect_white_cells_yolo(yolo_model_path, input_image_path, temp_csv)

    # Step 2: Crop white cells
    original_image, patches = crop_white_cells_from_csv(input_image_path, temp_csv)

    if len(patches) == 0:
        print(" No white cells detected!")
        return None

    # Step 3: Predict using Patch Classifier
    pred_class, probs = predict_patch_classifier(patch_classifier_model_path, original_image, patches)

    print(f" Predicted Class: {pred_class}")
    print(f" Class Probabilities: {probs}")

    os.remove(temp_csv)  # Clean temporary CSV

    return pred_class, probs
