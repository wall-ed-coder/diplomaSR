from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np

from utils.preprocessing import SIZES_FOR_CROPS
from utils.image_utils import open_image_RGB


class DataSetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


@dataclass
class CommonSRDataset(Dataset):
    csv_path: str
    root_to_data: str
    augmentation_for_SR: A.Compose
    augmentation_for_LR: A.Compose
    dataloader_kwargs: dict
    mode: DataSetMode = DataSetMode.VALIDATION
    length: Optional[int] = None
    csv_data: pd.DataFrame = None
    current_resize_shape: Tuple[int, int] = None

    def __post_init__(self):
        # self.csv_data = pd.read_csv(self.csv_path, sep=';')
        self.current_resize_shape = self.get_random_resize_shape()

    def get_random_resize_shape(self):
        return SIZES_FOR_CROPS[np.random.choice(len(SIZES_FOR_CROPS))]

    def get_dataloader(self) -> DataLoader:
        return DataLoader(self, **self.dataloader_kwargs, shuffle=(self.mode == DataSetMode.TRAIN))

    def __getitem__(self, index: int):
        img_name = self.csv_data.iloc[index]['imgName']
        img_path = os.path.join(self.root_to_data, img_name)

        original_image = open_image_RGB(img_path)
        preprocessed_SR_img = self.augmentation_for_SR(image=original_image)['image']

        preprocessed_LR_img = self.augmentation_for_LR(image=preprocessed_SR_img)['image']

        return {
            "lr_img": preprocessed_LR_img,
            "sr_img": preprocessed_SR_img,
        }

    def __len__(self):
        if self.mode == 'train' and self.length is not None:
            return self.length
        return self.csv_data.shape[0]

    def update(self):
        self.current_resize_shape = self.get_random_resize_shape()


@dataclass
class SRDatasets:
    train_dataset: CommonSRDataset
    val_dataset: CommonSRDataset
    test_dataset: CommonSRDataset
    train_loader: DataLoader = None
    val_loader: DataLoader = None
    test_loader: DataLoader = None

    def __post_init__(self):
        self.train_loader = self.train_loader or self.train_dataset.get_dataloader()
        self.val_loader = self.val_loader or self.val_dataset.get_dataloader()
        self.test_loader = self.test_loader or self.test_dataset.get_dataloader()

    def update(self):
        self.train_dataset.update()
        self.val_dataset.update()
        self.test_dataset.update()


if __name__ == '__main__':
    ds = CommonSRDataset(None, None, None, None, None, None, None, None, )

    dl = DataLoader(ds)

    print(ds.__dict__, dl.dataset.__dict__)

    ds.update()

    print(ds.__dict__, dl.dataset.__dict__)



