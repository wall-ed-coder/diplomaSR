from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from preprocessing.preprocessing import custom_resize_fn, preprocess_input
from preprocessing.preprocessing_for_lr import noise_and_blur_transforms, scale_transforms, resize_for_LR
from preprocessing.preprocessing_for_sr import rotate_and_shift_transforms, changing_structure_transforms, \
    specific_changing_color_transforms, usual_changing_color_transforms


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
            resize_shape: Tuple[int, int],
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
