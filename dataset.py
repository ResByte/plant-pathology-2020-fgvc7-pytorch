from typing import Callable
import pandas as pd
import cv2
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """Retrieves each data item for use with dataloaders"""

    def __init__(self,
                 img_root: str,
                 df: pd.DataFrame,
                 img_transforms:Callable = None,
                 is_train: bool = True
                 ):

        self.df = df
        self.img_root = img_root
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row['image_id']
        target = np.argmax(
            row[['healthy', 'multiple_diseases', 'rust', 'scab']].values)
#         label = np.zeros(4)
#         label[target] = 1.
        img = Image.open(os.path.join(
            self.img_root, filename+'.jpg')).convert('RGB')
        img = np.asarray(img)
        if self.img_transforms is not None:
            augmented = self.img_transforms(image=img)
            img = augmented['image']
        return img, target #np.asarray(label,dtype=np.float)