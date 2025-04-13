import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import argparse
import os

class WBCPredictor:
    def __init__(self, model_path='../models/wbc_classifier.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((320, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        with open('../models/class_mappings.json') as f:
            self.class_names = {v: k for k, v in json.load(f).items()}

    def _load_model(self, path):
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=None)  # Weights loaded from our trained model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
        return {
            'class': self.class_names[torch.argmax(probs).item()],
            'confidence': torch.max(probs).item(),
            'probabilities': {self.class_names[i]: p.item() 
                            for i, p in enumerate(probs.squeeze())}
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WBC Classification Predictor')
    parser.add_argument('--image', required=True, help='Path to blood cell image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        exit(1)
        
    predictor = WBCPredictor()
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2))