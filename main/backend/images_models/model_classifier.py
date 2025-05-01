import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class ModelClassifier:
    def __init__(self, model_path, image_size=(224, 224)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.image_size = image_size
        self.model = None
        self.class_names = [
            "blood",
            "brain",
            "chest",
            "skin"
        ]

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_model(self, num_classes):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_image(self, image_path):
        # Load the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Load the model and make predictions
        self.load_model(num_classes=4)  # Set the correct number of classes
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()

        # Map the predicted class index to the disease name
        pred_class_name = self.class_names[pred_class_idx]

        return pred_class_name, probs.cpu().numpy().flatten()

# How to use
# classifier = ModelClassifier(model_path=r"main\backend\models-weight\image-classifier-weight-mpdel.pth")
# pred_class_name, probs = classifier.predict_image(r"C:\Users\USER\Downloads\data\imageclassifier\train\blood\EOSINOPHIL___0_207.jpeg")
# print(f"Predicted class: {pred_class_name}")
# print(f"Class probabilities: {probs}")
