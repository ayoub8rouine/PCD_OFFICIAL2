import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class SkinModelClassifier:
    def __init__(self, model_path, num_classes=8, image_size=(224, 224)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.image_size = image_size
        self.model_path = model_path

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        base = models.resnet18(weights=None)
        features = nn.Sequential(*list(base.children())[:-1])
        classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        class ResNetClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = features
                self.classifier = classifier

            def forward(self, x):
                x = self.features(x).flatten(1)
                return self.classifier(x)

        return ResNetClassifier().to(self.device)

    def _load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return pred_class, probs.cpu().numpy().flatten()


# how to use 
# classifier = SkinModelClassifier("path/to/skin_model.pth", num_classes=8)
# pred_class, probs = classifier.predict("path/to/image.jpg")
# print(f"Predicted class: {pred_class}")
# print(f"Class probabilities: {probs}")
