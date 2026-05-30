import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import IP102Dataset
from src.utils import create_model, get_device, get_transforms, load_classes


def compute_metrics(confusion: torch.Tensor):
    diag = confusion.diag().to(torch.float32)
    precision = diag / confusion.sum(0).clamp(min=1).to(torch.float32)
    recall = diag / confusion.sum(1).clamp(min=1).to(torch.float32)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    precision = torch.nan_to_num(precision, nan=0.0)
    recall = torch.nan_to_num(recall, nan=0.0)
    f1 = torch.nan_to_num(f1, nan=0.0)
    accuracy = diag.sum() / confusion.sum().clamp(min=1).to(torch.float32)
    return (
        accuracy.item(),
        precision.mean().item(),
        recall.mean().item(),
        f1.mean().item(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/ip102_v1.1")
    parser.add_argument(
        "--classes-path",
        default="datasets/IP102_v1.1/Classification/classes.txt",
    )
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    split_file = data_root / f"{args.split}.txt"

    classes = load_classes(args.classes_path)
    if not classes:
        raise ValueError("No classes found")

    device = get_device()
    dataset = IP102Dataset(
        str(split_file), str(images_dir), transform=get_transforms(train=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(len(classes), pretrained=False)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu()
            for t, p in zip(labels, preds):
                confusion[t, p] += 1

    accuracy, precision, recall, f1 = compute_metrics(confusion)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")


if __name__ == "__main__":
    main()
