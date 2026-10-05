from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root = Path(root).resolve()
        self.rows = list(rows)
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root/row['path']).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Citra keluar dari folder data.')
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert('RGB')
        return self.transform(image),torch.tensor(float(row['age']),dtype=torch.float32),row['path']
