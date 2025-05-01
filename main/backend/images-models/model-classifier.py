import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

class ModelClassifier:
    def __init__(self, model_path, image_size=(224, 224), batch_size=32, num_workers=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = None
        self.class_names = None

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

    def predict_directory(self, test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        self.class_names = test_dataset.classes
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers)

        self.load_model(num_classes=len(self.class_names))

        all_preds = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        predicted_class_names = [self.class_names[p] for p in all_preds]
        return predicted_class_names


# how to use 
# classifier = ModelClassifier(model_path="path/to/model.pth")
# predictions = classifier.predict_directory("path/to/test/images")
# print(predictions)
