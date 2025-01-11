import os
import numpy as np
from torch.utils.data import Dataset
from typing import Optional,Callable
from PIL import Image

class ICCVDataset(Dataset):

    def __init__(self, 
        root : str,
        transform : Optional[Callable] = None,             
    ):
        super().__init__()

        self.root = root
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'images')
        self.labels_dir = os.path.join(self.root, 'labels')

        self.files = os.listdir(self.images_dir)
        self.files = map(lambda x: os.path.splitext(x)[0], self.files)
        self.files = list(self.files)

    def get_filename(self, idx : int) -> str:
        return self.files[idx]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx : int) -> dict:

        image_path = os.path.join(self.images_dir, self.files[idx] + '.jpg')
        label_path = os.path.join(self.labels_dir, self.files[idx] + '.regions.txt')

        image = Image.open(image_path).convert('RGB')

        with open(label_path, 'r') as f:
            label = f.readlines()

        label = [line.strip().split() for line in label]  
        label = [[int(x) for x in line] for line in label]
        label = np.array(label, dtype=np.uint8)

        data = {
            'image': image,
            'mask': label
        }

        if self.transform:
            data = self.transform(data)

        return data
