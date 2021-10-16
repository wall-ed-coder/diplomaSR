import numpy as np

import albumentations as A
import torch

DOWN_SCALE_COEF = [2, 4, 8, 16]
MAX_IMG_HEIGHT = 512
MAX_IMG_WIDTH = 512
MIN_IMG_HEIGHT = 32
MIN_IMG_WIDTH = 32
MAX_IMG_SIZE_IN_PIXELS: int = MAX_IMG_HEIGHT * MAX_IMG_WIDTH
SHIFT_LIMIT = 35
ASPECT_RATION = [
    (16, 9),
    (5, 4),
    (3, 2),
    (1, 1),
    (4, 3),
]

SIZES_FOR_CROPS = sorted(
    [
        # (16, 16), (32, 16), (32, 32), (64, 32), (64, 64),
        (128, 64), (128, 128), (256, 128), (256, 64), (256, 256)
        # (256, 256), (512, 384), (512, 512), (1024, 768), (1024, 1024)
    ]
)


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
    (height//scale_coef, width//scale_coef): A.Compose([
        A.Resize(height=height//scale_coef, width=width//scale_coef, always_apply=True)
    ]) for height, width in SIZES_FOR_CROPS
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
    # [(256, 256), (512, 288), (512, 384), (512, 512), (1024, 576), (1024, 768), (1024, 1024)]
    print(SIZES_FOR_CROPS)
