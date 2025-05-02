import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        attention = self.attention(x)
        x = x * attention
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x

class ChestClassifier:
    def __init__(self, model_path, num_classes=4, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomResNet(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

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

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            pred_class_idx = torch.argmax(outputs, dim=1).item()

        return self.class_labels[pred_class_idx]

# === USAGE EXAMPLE ===
#if __name__ == '__main__':
    #classifier = ChestClassifier(model_path=r'main\backend\models-weight\chest-wright-model.pth')
    #prediction = classifier.predict(r'main\backend\images_models\n8.jpg')
    #print("Prediction:", prediction)



# how the chest model work
# # Load and use the ChestModel
# classifier = ChestClassifier(model_path='models/chest_model.pth')
# prediction = classifier.predict('images/test_xray.jpg')
# print("Prediction:", prediction)
