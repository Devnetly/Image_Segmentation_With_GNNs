import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional

class LazyNumpyArrayReader(Dataset):

    def __init__(self, root : str):
        """
            A class to read large .npy files without loading them into memory.

            Args :
            - root : str : Path to the .npy file.

            Returns:
            - None
        """


        super().__init__()

        self.root = root
        self.metadata = self.get_npy_metadata(root)
        self.memmap = np.memmap(root, 
            dtype=self.metadata["dtype"], 
            mode='r', 
            shape=self.metadata["shape"], 
            order=('F' if self.metadata["fortran_order"] else 'C')
        )

    
    def get_npy_metadata(self, file_path) -> dict:
        """
        Get the shape and data type of a large .npy file without loading it into memory.

        Args:
            - file_path : str : Path to the .npy file.

        Returns:
            - dict : A dictionary containing the shape and data type of the array.
        """

        with open(file_path, 'rb') as f:
            # Read the version and header
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        
        return {
            "shape": shape,
            "dtype": dtype,
            "fortran_order": fortran_order,
        }

    def __len__(self) -> int:
        return self.metadata['shape'][0]

    def __getitem__(self, idx : int) -> np.ndarray:
        return self.memmap[idx]

class PanNukeDatasset(Dataset):

    def __init__(self, root : str):
        super().__init__()

        self.root = root

        self.fold_name = ''.join(os.path.basename(root).split(' ')).lower()

        self.images_file = os.path.join(root, "images", self.fold_name, "images.npy")
        self.types_file = os.path.join(root, "images", self.fold_name, "types.npy")
        self.masks_file = os.path.join(root, "masks", self.fold_name, "masks.npy")

        self.images = LazyNumpyArrayReader(self.images_file)
        self.types = np.load(self.types_file)
        self.masks = LazyNumpyArrayReader(self.masks_file)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx : int) -> dict:
        
        image = self.images[idx]
        type = self.types[idx]
        mask = self.masks[idx]

        image = image.clip(0, 255).astype(np.uint8)
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        image = np.stack((G, B, R), axis=-1)

        return {
            "image": image,
            "type": type,
            "mask": mask
        }


class ISICDataset(Dataset):

    def __init__(self, 
        root : str,
        img_transform : Optional[Callable] = None,
        mask_transform : Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.files = self._get_files()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

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

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask
        }