import argparse

import torch

from src.utils import create_model, get_device, get_transforms, load_classes, load_image, predict_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument(
        "--classes-path",
        default="datasets/IP102_v1.1/Classification/classes.txt",
    )
    args = parser.parse_args()

    classes = load_classes(args.classes_path)
    if not classes:
        raise ValueError("No classes found")

    device = get_device()
    model = create_model(len(classes), pretrained=False)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    image = load_image(args.image)
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0)

    idx, conf = predict_tensor(model, tensor, device)
    print(classes[idx])
    print(f"confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
