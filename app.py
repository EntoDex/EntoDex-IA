from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.utils import create_model, get_device, get_transforms, load_classes, predict_tensor


@st.cache_resource
def load_model(model_path: str, classes_path: str):
    classes = load_classes(classes_path)
    device = get_device()
    model = create_model(len(classes), pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes, device


def main():
    st.set_page_config(page_title="EntoDex-IA", page_icon="")
    st.title("EntoDex-IA")

    model_path = "models/best_model.pth"
    classes_path = "datasets/IP102_v1.1/Classification/classes.txt"

    if not Path(model_path).exists():
        st.error("Model not found. Train the model first.")
        return

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    model, classes, device = load_model(model_path, classes_path)
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0)
    idx, conf = predict_tensor(model, tensor, device)

    st.write(f"Class: {classes[idx]}")
    st.write(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
