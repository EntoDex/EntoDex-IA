import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import IP102Dataset
from src.utils import create_model, get_device, get_transforms, load_classes


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    scaler=None,
    amp: bool = False,
    grad_clip: float = 0.0,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    autocast = torch.cuda.amp.autocast if amp else nullcontext

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                if amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total if total else 0.0
    avg_acc = correct / total if total else 0.0
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/ip102_v1.1")
    parser.add_argument(
        "--classes-path",
        default="datasets/IP102_v1.1/Classification/classes.txt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", default="models/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    train_file = data_root / "train.txt"
    val_file = data_root / "val.txt"

    classes = load_classes(args.classes_path)
    if not classes:
        raise ValueError("No classes found")

    device = get_device()
    train_ds = IP102Dataset(
        str(train_file), str(images_dir), transform=get_transforms(train=True)
    )
    val_ds = IP102Dataset(
        str(val_file), str(images_dir), transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = create_model(len(classes), pretrained=True).to(device)
    if args.class_weights:
        counts = torch.zeros(len(classes), dtype=torch.float32)
        for _, label in train_ds.samples:
            if 0 <= label < len(classes):
                counts[label] += 1
        weights = counts.sum() / counts.clamp(min=1)
        weights = weights / weights.mean()
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float("inf")
    patience_counter = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            scaler=scaler,
            amp=use_amp,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {lr:.2e}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping")
                break

    print(f"Best model saved to {output_path}")


if __name__ == "__main__":
    main()
