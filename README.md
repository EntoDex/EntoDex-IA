EntoDex-IA

Prerequisites
- Python

Quick setup (venv)
- bash setup_env.sh

Manual setup
- python -m venv .venv
- source .venv/bin/activate
- python -m pip install --upgrade pip
- pip install -r requirements.txt

Dataset layout
- images: datasets/ip102_v1.1/images
- splits: datasets/ip102_v1.1/train.txt, val.txt, test.txt
- classes: datasets/IP102_v1.1/Classification/classes.txt

Train
- python -m src.train --epochs 10 --batch-size 32 --lr 1e-4 --patience 3

Evaluate
- python -m src.evaluate --split test

Predict one image
- python -m src.predict --image datasets/ip102_v1.1/images/00002.jpg

Streamlit app
- streamlit run app.py

Notes
- For CUDA, install the PyTorch build matching your GPU from https://pytorch.org/get-started/locally/
