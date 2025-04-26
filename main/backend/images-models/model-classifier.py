import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

def test_image_classifier(test_dir, model_path, batch_size=32, num_workers=4):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_names = test_dataset.classes

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    predicted_class_names = [class_names[p] for p in all_preds]
    return predicted_class_names
