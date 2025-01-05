import os
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional

class ISICDataset(Dataset):

    def __init__(self, 
        root : str,
        img_transform : Optional[Callable] = None,
        mask_transform : Optional[Callable] = None,
        return_mask : bool = False,
    ) -> None:
        super().__init__()

        self.root = root
        self.files = self._get_files()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.return_mask = return_mask

    def _get_files(self) -> list[str]:

        files = os.listdir(self.root)
        files = [os.path.splitext(file)[0].replace("_Segmentation","") for file in files]
        files = list(set(files))
        files = sorted(files)

        return files
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx : int) -> dict:

        filename = self.files[idx]
        img_path = os.path.join(self.root, f"{filename}.jpg")
        mask_path = os.path.join(self.root, f"{filename}_Segmentation.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        data = {
            'image': image,
            'mask': mask
        }

        if self.img_transform:
            data['image'] = self.img_transform(data['image'])

        if self.mask_transform and self.return_mask:
            data['mask'] = self.mask_transform(data['mask'])

        return data