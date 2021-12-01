import cv2
import numpy as np

import albumentations as A
import torch

DOWN_SCALE_COEF = [2, 4, 8, 16]
MAX_IMG_HEIGHT = 512
MAX_IMG_WIDTH = 512
MIN_IMG_HEIGHT = 64
MIN_IMG_WIDTH = 64
MAX_IMG_SIZE_IN_PIXELS: int = MAX_IMG_HEIGHT * MAX_IMG_WIDTH
SHIFT_LIMIT = 35
ASPECT_RATION = [
    (4, 1),
    (2, 1),
    (1, 1),
]

SIZES_FOR_CROPS = tuple(sorted(filter(
    lambda t: MIN_IMG_HEIGHT <= t[0] <= MAX_IMG_HEIGHT and MIN_IMG_WIDTH <= t[1] <= MAX_IMG_WIDTH,
    [
        (2**i//asp_ration[1], 2**i//asp_ration[0])
        for i in range(6, 10)
        for asp_ration in ASPECT_RATION
    ]
)))


RANDOM_RESIZE_AUGMENTATIONS = {
    (height, width): A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True),
        A.OneOf([
            A.RandomCrop(height=height, width=width, p=0.8),
            A.RandomResizedCrop(height=height, width=width, p=0.2),
        ], p=1.)
    ]) for height, width in SIZES_FOR_CROPS
}


# todo добавить разную интерполяцию
RESIZE_SCALE_DOWN_LR = {
    (height//scale_coef, width//scale_coef): A.OneOf([
        A.Resize(height=height//scale_coef, width=width//scale_coef, interpolation=cv2.INTER_LINEAR,),
        A.Resize(height=height//scale_coef, width=width//scale_coef, interpolation=cv2.INTER_NEAREST,),
        A.Resize(height=height//scale_coef, width=width//scale_coef, interpolation=cv2.INTER_CUBIC,),
        A.Resize(height=height//scale_coef, width=width//scale_coef, interpolation=cv2.INTER_AREA,),
    ], p=1.) for height, width in SIZES_FOR_CROPS
    for scale_coef in DOWN_SCALE_COEF
}


def preprocess_input(image: np.array):
    image = (image / 255.).astype('float32').transpose((2, 0, 1))
    return torch.from_numpy(image)


def custom_resize_fn(image, resize_shape=None):
    if resize_shape is None:
        random_idx = np.random.choice(len(SIZES_FOR_CROPS))
        resize_shape = SIZES_FOR_CROPS[random_idx]

    return RANDOM_RESIZE_AUGMENTATIONS[resize_shape](image=image)['image']


# todo исправить баг что постоянно появляются обрезанные какие-то полоски черные + на них даже
#  накладывается шум что тоже плохо
#  будто изображение реежется и конкатится постоянно
#  блюр слишком сильный для маленьких картинок


if __name__ == '__main__':
    print(SIZES_FOR_CROPS)
