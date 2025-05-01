import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import os
from tqdm import tqdm

class BloodModel:
    def __init__(self, yolo_model_path, patch_classifier_model_path, num_classes=4, device=None):
        self.yolo_model_path = yolo_model_path
        self.patch_classifier_model_path = patch_classifier_model_path
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.yolo_model = self._load_yolo_model()
        self.patch_classifier = self._load_patch_classifier()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # === Internal: Load YOLOv5 model ===
    def _load_yolo_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=self.yolo_model_path, force_reload=True)

    # === Internal: Load PatchClassifier ===
    def _load_patch_classifier(self):
        model = self.PatchClassifier(num_classes=self.num_classes).to(self.device)
        model.load_state_dict(torch.load(self.patch_classifier_model_path, map_location=self.device))
        model.eval()
        return model

    # === Internal: Detect white blood cells ===
    def detect_white_cells(self, image_path, output_csv_path):
        img = Image.open(image_path).convert("RGB")
        results = self.yolo_model(img, size=640)
        detections = results.pandas().xyxy[0]

        csv_data = []
        for _, row in detections.iterrows():
            csv_data.append({
                "class": int(row['class']),
                "xmin": int(row['xmin']),
                "xmax": int(row['xmax']),
                "ymin": int(row['ymin']),
                "ymax": int(row['ymax']),
            })

        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv_path, index=False)
        return img, df

    # === Internal: Crop detected patches ===
    def crop_white_cells(self, image, detections_csv):
        df = pd.read_csv(detections_csv)
        white_cell_crops = []

        for _, row in df.iterrows():
            if row['class'] == 1:  # Adjust if other class IDs also represent white cells
                crop = image.crop((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
                white_cell_crops.append(crop)

        return white_cell_crops

    # === Public: Predict class for given image path ===
    def predict(self, image_path):
        temp_csv = "temp_detections.csv"
        original_image, _ = self.detect_white_cells(image_path, temp_csv)
        patches = self.crop_white_cells(original_image, temp_csv)

        if len(patches) == 0:
            os.remove(temp_csv)
            print("No white cells detected!")
            return None, None

        with torch.no_grad():
            orig_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
            patch_tensors = [self.transform(p) for p in patches]
            patch_tensor_batch = torch.stack(patch_tensors).unsqueeze(0).to(self.device)

            outputs = self.patch_classifier(orig_tensor, [patch_tensor_batch])
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        os.remove(temp_csv)
        return pred_class, probs.cpu().numpy()

    # === Patch Classifier (inner class) ===
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
        
#how to use
# # Instantiate the model
# model = BloodModel(
#     yolo_model_path="best_yolov5_model.pt",
#     patch_classifier_model_path="blood_patch_classifier.pth",
#     num_classes=4  # or whatever number of WBC types you're classifying
# )

# # Predict
# pred_class, class_probs = model.predict("sample_wbc_image.jpg")

# print(f"Predicted Class: {pred_class}")
# print(f"Class Probabilities: {class_probs}")

