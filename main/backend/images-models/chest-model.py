import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ChestClassifier:
    def __init__(self, model_path, num_classes=4, device=None):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = [
            'adenocarcinoma_left.lower.lobe_T2',
            'large.cell.carcinoma_left.hilum_T2',
            'normal',
            'squamous.cell.carcinoma_left.hilum'
        ]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        model = models.resnet50(pretrained=True)

        # Freeze all layers except layer4
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Replace classifier head
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        return model.to(self.device)

    def _load_weights(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Feature extraction
        with torch.no_grad():
            x = self.model.conv1(img_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            # Apply attention
            attn_map = self.attention(x)
            x = x * attn_map

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)

            pred_class_idx = torch.argmax(x, dim=1).item()

        return self.class_labels[pred_class_idx]


# how the chest model work
# # Load and use the ChestModel
# classifier = ChestClassifier(model_path='models/chest_model.pth')
# prediction = classifier.predict('images/test_xray.jpg')
# print("Prediction:", prediction)
