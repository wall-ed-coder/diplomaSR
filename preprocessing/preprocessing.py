from typing import Tuple, Optional
from dataclasses import dataclass
import albumentations as A
import numpy as np


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


def noise_and_blur_transforms()->A.Compose:
    blur_transforms = [
        A.OneOf([
            *[
                A.RandomFog(fog_coef_lower=i, fog_coef_upper=i, alpha_coef=0., p=0.05)
                for i in [0.05, 0.1, 0.15, 0.2]
            ],
            *[
                A.Blur(i, p=0.05)
                for i in range(3, 6)
            ],
            A.GlassBlur(sigma=0.05, max_delta=1, iterations=1, p=0.05),
            A.MedianBlur(p=0.05),
            A.MotionBlur(p=0.05),
            A.GaussianBlur(p=0.05),

        ], p=0.5),
    ]

    noise_transforms = [
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.ISONoise(p=1.),
            A.MultiplicativeNoise(p=0.2),
        ], p=0.5),
    ]
    return A.Compose(blur_transforms + noise_transforms)


def custom_resize_fn(image, resize_shape=None):
    if resize_shape is None:
        random_idx = np.random.choice(len(SIZES_FOR_CROPS))
        resize_shape = SIZES_FOR_CROPS[random_idx]

    return RANDOM_RESIZE_AUGMENTATIONS[resize_shape](image=image)['image']


def scale_transforms() -> A.Compose:
    scale_transforms = [
        A.OneOf([
            A.Downscale(scale_min=0.4, scale_max=0.9, p=0.05),
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.1),
        ], p=0.05),

    ]
    return A.Compose(scale_transforms)


def rotate_and_shift_transforms()->A.Compose:
    rotate_transforms = [
        A.OneOf([
            A.Rotate(limit=45, p=1.),  # will be like a mirror
            # A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
            A.Affine(p=0.8),
            A.PiecewiseAffine(p=0.1),
        ], p=0.2),
    ]

    shift_transforms = [
        A.RGBShift(r_shift_limit=SHIFT_LIMIT, b_shift_limit=SHIFT_LIMIT, g_shift_limit=SHIFT_LIMIT, p=0.05),
    ]

    return A.Compose(shift_transforms + rotate_transforms)


def usual_changing_color_transforms()->A.Compose:
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.1),
    ]
    return A.Compose(transforms)


def specific_changing_color_transforms()->A.Compose:
    transforms = [
        A.SomeOf([
            A.CLAHE(p=0.5),
            A.ColorJitter(p=0.5),
            A.ChannelDropout(p=0.1),
            A.ToSepia(p=0.1),
            A.Posterize(p=0.1),
            A.Sharpen(p=0.1),
            A.Solarize(p=0.1),
            A.ChannelShuffle(p=0.05),
            A.Equalize(p=0.1),
            A.RandomToneCurve(scale=0.3, p=0.1),
            A.FancyPCA(alpha=0.2, p=0.1),
            A.FancyPCA(alpha=0.3, p=0.1),
            A.FancyPCA(alpha=0.4, p=0.1),
            A.FancyPCA(alpha=0.5, p=0.1),
            A.Emboss(alpha=(0.2, 1.), p=0.25),
            A.ElasticTransform(p=0.1),
            A.HueSaturationValue(p=0.1),
        ], n=2, p=0.07),
    ]
    return A.Compose(transforms)


def changing_structure_transforms() -> A.Compose:
    transforms = [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.CoarseDropout(p=1.),
            A.GridDropout(
                shift_x=50, shift_y=50, holes_number_x=10, holes_number_y=10, unit_size_min=5, unit_size_max=15, p=0.005
            ),
            A.RandomGridShuffle(grid=(3, 6), p=0.1),
            A.RandomGridShuffle(grid=(4, 4), p=0.1),
            A.RandomGridShuffle(grid=(5, 4), p=0.1),
            A.RandomGridShuffle(grid=(3, 8), p=0.1),
            A.RandomGridShuffle(grid=(8, 8), p=0.1),
            A.RandomGridShuffle(grid=(6, 8), p=0.1),
            A.RandomGridShuffle(grid=(8, 16), p=0.1),
        ], p=0.05),

    ]
    return A.Compose(transforms)


def resize_for_LR(image, resize_shape):
    return RESIZE_SCALE_DOWN_LR[resize_shape](image=image)['image']


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

