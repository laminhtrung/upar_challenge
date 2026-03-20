import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiFolderDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None, skiprows=1):
        self.data = pd.read_csv(csv_file, skiprows=skiprows)
        self.img_dirs = img_dirs
        self.transform = transform
        self.samples = []

        for idx in range(len(self.data)):
            img_name = os.path.basename(str(self.data.iloc[idx, 0]))
            img_path = None

            for img_dir in self.img_dirs:
                candidate = os.path.join(img_dir, img_name)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            labels = self.data.iloc[idx, 1:-1].values.astype(float)
            self.samples.append((img_path, labels))

        print(f"[Dataset] Loaded {len(self.samples)} valid samples from {csv_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image, labels