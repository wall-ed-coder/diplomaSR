import numpy as np

import albumentations as A

DOWN_SCALE_COEF = [4, 8, 16]
MAX_IMG_HEIGHT = 1024
MAX_IMG_WIDTH = 1024
MIN_IMG_HEIGHT = 224
MIN_IMG_WIDTH = 224
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
    filter(
        lambda x: x[0] % 8 == 0 and x[1] % 8 == 0 and x[0] >= MIN_IMG_HEIGHT and x[1] >= MIN_IMG_WIDTH, [
            (
                (
                    MAX_IMG_HEIGHT // i // 1,
                    int(MAX_IMG_WIDTH // i * (aspect_ration_width / aspect_ration_height)) // 1
                )
            ) for i in range(1, 6)
            for aspect_ration_height, aspect_ration_width in ASPECT_RATION
        ]
    )
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
    return image


def custom_resize_fn(image, resize_shape=None):
    if resize_shape is None:
        random_idx = np.random.choice(len(SIZES_FOR_CROPS))
        resize_shape = SIZES_FOR_CROPS[random_idx]

    return RANDOM_RESIZE_AUGMENTATIONS[resize_shape](image=image)['image']


# todo исправить баг что постоянно появляются обрезанные какие-то полоски черные + на них даже
#  накладывается шум что тоже плохо
#  будто изображение реежется и конкатится постоянно
#  блюр слишком сильный для маленьких картинок

