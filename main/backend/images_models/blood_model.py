import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import os
from ultralytics import YOLO
from tqdm import tqdm

class BloodModel:
    def __init__(self, yolo_model_path, patch_classifier_model_path, num_classes=4, device=None, class_names=None):
        self.yolo_model_path = yolo_model_path
        self.patch_classifier_model_path = patch_classifier_model_path
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If class_names isn't passed, default to the specified classes
        self.class_names = class_names if class_names else ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
        
        self.yolo_model = self._load_yolo_model()
        self.patch_classifier = self._load_patch_classifier()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_yolo_model(self):
        return YOLO(self.yolo_model_path)

    def _load_patch_classifier(self):
        model = self.PatchClassifier(num_classes=self.num_classes).to(self.device)
        model.load_state_dict(torch.load(self.patch_classifier_model_path, map_location=self.device))
        model.eval()
        return model

    def detect_white_cells(self, image_path, output_csv_path):
        results = self.yolo_model.predict(image_path, conf=0.5, device=0 if self.device.type == "cuda" else "cpu")
        boxes_matrix = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            boxes_matrix.append({
                "class": cls,
                "xmin": int(xmin),
                "xmax": int(xmax),
                "ymin": int(ymin),
                "ymax": int(ymax),
            })

        df = pd.DataFrame(boxes_matrix)
        df.to_csv(output_csv_path, index=False)

        image = Image.open(image_path).convert("RGB")
        return image, df

    def crop_white_cells(self, image, detections_csv):
        df = pd.read_csv(detections_csv)
        white_cell_crops = []

        for _, row in df.iterrows():
            if row['class'] == 1:  # You can adjust this based on which class means white cell
                crop = image.crop((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
                white_cell_crops.append(crop)

        return white_cell_crops

    def predict(self, image_path):
        temp_csv = "temp_detections.csv"
        original_image, _ = self.detect_white_cells(image_path, temp_csv)
        patches = self.crop_white_cells(original_image, temp_csv)

        if len(patches) == 0:
            os.remove(temp_csv)
            print("No white cells detected!")
            return None, None

        # Debugging: Print shape of the patches and original image tensor
        with torch.no_grad():
            orig_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

            # Apply the transform to each patch and ensure they are stacked correctly
            patch_tensors = [self.transform(p) for p in patches]
            print(f"Shape of each patch after transformation: {patch_tensors[0].shape}")  # Check the shape of the first patch

            # Stack patches properly: each patch should have shape [C, H, W], so stacking them results in [N, C, H, W]
            patch_tensor_batch = torch.stack(patch_tensors).to(self.device)
            print(f"Shape of stacked patch tensor: {patch_tensor_batch.shape}")  # Check the batch shape

            # Ensure we have the right input shape
            if patch_tensor_batch.ndimension() == 3:  # Single patch, add batch dimension
                patch_tensor_batch = patch_tensor_batch.unsqueeze(0)  # Shape: [1, C, H, W]
            
            # Forward pass through the patch classifier
            outputs = self.patch_classifier(orig_tensor, [patch_tensor_batch])
            probs = torch.softmax(outputs, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()

            # Get the predicted class name from the class_names list
            pred_class_name = self.class_names[pred_class_idx]

        os.remove(temp_csv)
        return pred_class_name, probs.cpu().numpy()


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

            # Ensure original features retain batch dimension
            orig_feats = self.orig_features(originals.to(device)).squeeze(-1).squeeze(-1)  # shape: [B, 512]

            for i, patches in enumerate(patch_lists):
                patches = patches.to(device)

                # Check the shape of patches
                print(f"patches shape before resnet: {patches.shape}")

                # Ensure that patches are treated as a batch
                if patches.ndimension() == 3:  # Single patch with shape [C, H, W]
                    patches = patches.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
                
                feats = self.patch_features(patches).squeeze(-1).squeeze(-1)  # shape: [N, 512]
                if feats.ndim == 1:
                    feats = feats.unsqueeze(0)

                weights = self.attention(feats)  # shape: [N, 1]
                weighted_feat = (feats * weights).sum(dim=0)  # shape: [512]

                combined = torch.cat([orig_feats[i].view(-1), weighted_feat.view(-1)], dim=0)  # shape: [1024]
                batch_features.append(combined)

            return self.classifier(torch.stack(batch_features))


#how to use 
# model = BloodModel(
#     yolo_model_path=r'main\backend\models-weight\yolo-weight-model.pt',
#     patch_classifier_model_path=r'main\backend\models-weight\blood-weight-model.pth',
#     num_classes=4
# )

# pred_class, class_probs = model.predict(
#     r"C:\Users\USER\Downloads\data\project - yolo\data\dataset2-master\dataset2-master\images\TRAIN\MONOCYTE\_0_180.jpeg"
# )

# print(f"Predicted Class: {pred_class}")
# print(f"Class Probabilities: {class_probs}")
