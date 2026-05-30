from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class IP102Dataset(Dataset):
    def __init__(self, split_file: str, images_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename = parts[0]
                label = int(parts[1])
                self.samples.append((filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, label = self.samples[idx]
        path = self.images_dir / filename
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
