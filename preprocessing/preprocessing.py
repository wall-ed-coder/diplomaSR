from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from preprocessing.preprocessing_for_lr import noise_and_blur_transforms, scale_transforms, \
    resize_for_LR
from preprocessing.preprocessing_for_sr import rotate_and_shift_transforms, usual_changing_color_transforms, \
    specific_changing_color_transforms, changing_structure_transforms, RANDOM_RESIZE_AUGMENTATIONS

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


def preprocess_input(image: np.array):
    image = (image / 255.).astype('float32').transpose((2, 0, 1))
    return image


def custom_resize_fn(image, resize_shape=None):
    if resize_shape is None:
        random_idx = np.random.choice(len(SIZES_FOR_CROPS))
        resize_shape = SIZES_FOR_CROPS[random_idx]

    return RANDOM_RESIZE_AUGMENTATIONS[resize_shape](image=image)['image']


# todo изменить название на нормальное
@dataclass
class ApplyAlbumentation:
    add_specific_changing_color_transforms: bool = True
    add_usual_changing_color_transforms: bool = True
    add_changing_structure_transforms: bool = True
    prob_do_nothing: float = 0.02

    rotate_and_shift = None
    changing_structure = None
    specific_changing_color = None
    usual_changing_color = None
    noise_and_blur = None
    scale = None

    def __post_init__(self):
        self.rotate_and_shift = rotate_and_shift_transforms()

        if self.add_changing_structure_transforms:
            self.changing_structure = changing_structure_transforms()

        if self.add_specific_changing_color_transforms:
            self.specific_changing_color = specific_changing_color_transforms()

        if self.add_usual_changing_color_transforms:
            self.usual_changing_color = usual_changing_color_transforms()

        self.noise_and_blur = noise_and_blur_transforms()
        self.scale = scale_transforms()

    def apply_lr_transform(
            self,
            image: np.array,
            resize_shape: Optional[Tuple[int, int]],
    ) -> np.array:
        image = resize_for_LR(image, resize_shape)
        if self.noise_and_blur:
            image = self.noise_and_blur(image=image)['image']

        if self.scale:
            image = self.scale(image=image)['image']

        return image

    def apply_sr_transform(
            self,
            image: np.array,
            resize_shape: Optional[Tuple[int, int]] = None,
    ) -> np.array:
        rand_num = np.random.rand()

        if self.rotate_and_shift and rand_num > self.prob_do_nothing:
            image = self.rotate_and_shift(image=image)['image']

        image = custom_resize_fn(image, resize_shape)
        if rand_num > self.prob_do_nothing:
            if self.changing_structure:
                image = self.changing_structure(image=image)['image']
            if self.specific_changing_color:
                image = self.specific_changing_color(image=image)['image']
            if self.usual_changing_color:
                image = self.usual_changing_color(image=image)['image']

        return image

    def apply_transpose_and_standardization(self, image: np.array,) -> np.array:
        return preprocess_input(image)


# todo исправить баг что постоянно появляются обрезанные какие-то полоски черные + на них даже
#  накладывается шум что тоже плохо
#  будто изображение реежется и конкатится постоянно
#  блюр слишком сильный для маленьких картинок

