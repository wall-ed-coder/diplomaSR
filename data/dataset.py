from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

from preprocessing.preprocessing import SIZES_FOR_CROPS
from preprocessing.apply_albumentations import ApplyAlbumentation
from utils.image_utils import open_image_RGB


class DataSetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


@dataclass
class CommonSRDataset(Dataset):
    csv_path: str
    root_to_data: str
    augmentation: ApplyAlbumentation
    scale_coef: int
    dataloader_kwargs: dict
    mode: DataSetMode = DataSetMode.VALIDATION
    length: Optional[int] = None

    csv_data: pd.DataFrame = None
    current_resize_shape: Tuple[int, int] = None
    current_resize_shape_lr: Tuple[int, int] = None

    def __post_init__(self):
        self.csv_data = pd.read_csv(self.csv_path, sep=';')
        self.current_resize_shape = self.get_random_resize_shape()
        self.current_resize_shape_lr = (
            self.current_resize_shape[0]//self.scale_coef, self.current_resize_shape[1]//self.scale_coef,
        )

    def get_random_resize_shape(self):
        return SIZES_FOR_CROPS[np.random.choice(len(SIZES_FOR_CROPS))]

    def get_dataloader(self) -> DataLoader:
        if 'shuffle' not in self.dataloader_kwargs:
            self.dataloader_kwargs.update(shuffle=(self.mode == DataSetMode.TRAIN))
        return DataLoader(self, **self.dataloader_kwargs)

    def __getitem__(self, index: int):
        img_name = self.csv_data.iloc[index]['imgName']
        img_path = os.path.join(self.root_to_data, img_name)
        original_image = open_image_RGB(img_path)
        preprocessed_SR_img = self.augmentation.apply_sr_transform(
            image=original_image, resize_shape=self.current_resize_shape
        )

        preprocessed_LR_img = self.augmentation.apply_lr_transform(
            image=preprocessed_SR_img, resize_shape=self.current_resize_shape_lr
        )

        return {
            "lr_img": self.augmentation.apply_transpose_and_standardization(preprocessed_LR_img),
            "sr_img": self.augmentation.apply_transpose_and_standardization(preprocessed_SR_img),
        }

    def __len__(self):
        if self.mode == 'train' and self.length is not None:
            return self.length
        return self.csv_data.shape[0]

    def update(self):
        self.current_resize_shape = self.get_random_resize_shape()
        self.current_resize_shape_lr = (
            self.current_resize_shape[0]//self.scale_coef, self.current_resize_shape[1]//self.scale_coef,
        )


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
    from utils import visualize_img_from_array
    transforms = ApplyAlbumentation()

    ds = CommonSRDataset(
        csv_path='/Users/nikita/Desktop/diploma_sr2/test_df.csv',
        root_to_data='/Users/nikita/Downloads/',
        scale_coef=4,
        augmentation=transforms,
        dataloader_kwargs={'batch_size': 4, 'num_workers': 4},
    )

    dl = DataLoader(ds)
    for epoch in range(3):
        for batch in dl:
            for sr_img, lr_img in zip(batch['sr_img'], batch['lr_img']):
                visualize_img_from_array(sr_img)
                visualize_img_from_array(lr_img)
                # print(sr_img.shape, lr_img.shape)
        ds.update()




