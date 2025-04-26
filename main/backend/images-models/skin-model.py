import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === MODEL ===
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

# === TRANSFORM ===
def get_test_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# === PREDICTION FUNCTION ===
def predict_skin_image(model_path, image_input, num_classes=8, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ResNetClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Handle input (path or PIL Image)
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input

    transform = get_test_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return pred_class, probs.cpu().numpy().flatten()
