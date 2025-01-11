import os
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional,Callable


class EMD6Dataset(Dataset):

    def __init__(self, 
        root : str,
        transform : Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.root = root

        self.masks_folder = os.path.join(root,'EMDS5-Ground Truth')
        self.images_folder = os.path.join(root,'EMDS5-Original')
        self.transform = transform

        self.files = self._get_files()

    def get_filename(self, idx : int) -> str:
        return os.path.basename(self.files[idx][0])
    
    def _get_files(self) -> list[tuple[str,str,int]]:
        
        classes = os.listdir(self.masks_folder)

        files = []

        for class_ in classes:

            mask_class_folder = os.path.join(self.masks_folder,class_)
            image_class_folder = os.path.join(self.images_folder,class_)

            masks = sorted(os.listdir(mask_class_folder))
            images = sorted(os.listdir(image_class_folder))

            for mask,image in zip(masks,images):

                mask_path = os.path.join(mask_class_folder,mask)
                image_path = os.path.join(image_class_folder,image)

                files.append((image_path,mask_path,class_))

        return files
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx : int) -> dict:

        image_path, mask_path, class_ = self.files[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        data =  {
            'image' : image,
            'mask' : mask,
            'class' : class_
        }

        if self.transform:
            data = self.transform(data)

        return data
