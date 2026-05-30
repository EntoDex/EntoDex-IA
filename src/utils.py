from typing import List

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


def load_classes(classes_path: str) -> List[str]:
    items = []
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = int(parts[0])
            name = " ".join(parts[1:]).strip()
            items.append((idx, name))
    if not items:
        return []
    items.sort(key=lambda x: x[0])
    min_idx = items[0][0]
    max_idx = items[-1][0]
    offset = 1 if min_idx == 1 else 0
    class_names = [""] * (max_idx - offset + 1)
    for idx, name in items:
        pos = idx - offset
        if 0 <= pos < len(class_names):
            class_names[pos] = name
    for i, name in enumerate(class_names):
        if not name:
            class_names[i] = f"class_{i}"
    return class_names


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def create_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def predict_tensor(model: torch.nn.Module, image_tensor: torch.Tensor, device):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()
