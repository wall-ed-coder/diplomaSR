from typing import Tuple, Optional

import albumentations as A
import cv2
import numpy as np


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

RESIZE_AUGMENTATIONS = {
    (height, width): A.Compose([
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True),
        A.OneOf([
            A.RandomCrop(height=height, width=width, p=0.8),
            A.RandomResizedCrop(height=height, width=width, p=0.2),
        ], p=1.)
    ]) for height, width in SIZES_FOR_CROPS
}


def preprocess_input(image: np.array):
    image = (image / 255.).transpose(2, 0, 1).astype('float32')
    return image


def noise_and_blur_transforms():
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
            A.GlassBlur(sigma=0.1, max_delta=1, iterations=1, p=0.05),
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
    return blur_transforms + noise_transforms


def custom_resize_fn(image, resize_shape: Optional[Tuple[int, int]] = None):

    if resize_shape is None:
        random_idx = np.random.choice(len(SIZES_FOR_CROPS))
        resize_shape = SIZES_FOR_CROPS[random_idx]

    return RESIZE_AUGMENTATIONS[resize_shape](image=image)['image']


def rotate_and_scale_and_shift_transforms():
    rotate_transforms = [
        A.OneOf([
            A.Rotate(limit=45, p=1.),  # will be like a mirror
            A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
            A.Affine(p=1.),
            A.PiecewiseAffine(p=0.25),
        ], p=0.2),
    ]
    scale_transforms = [
        A.Downscale(scale_min=0.4, scale_max=0.9, p=0.05),
    ]

    shift_transforms = [
        A.RGBShift(r_shift_limit=SHIFT_LIMIT, b_shift_limit=SHIFT_LIMIT, g_shift_limit=SHIFT_LIMIT, p=0.05),
    ]

    return shift_transforms + scale_transforms + rotate_transforms


def usual_changing_color_transforms():
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.1),
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.2),
    ]
    return transforms


def specific_changing_color_transforms():
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
        ], n=2, p=0.05),
    ]
    return transforms


def changing_structure_transforms():
    transforms = [
        A.OneOf([
            A.CoarseDropout(p=1.),
            A.GridDropout(
                shift_x=50, shift_y=50, holes_number_x=10, holes_number_y=10, unit_size_min=5, unit_size_max=15, p=0.005
            ),
            A.RandomGridShuffle(grid=(3, 6), p=0.1),
            A.RandomGridShuffle(grid=(4, 4), p=0.1),
            A.RandomGridShuffle(grid=(5, 4), p=0.1),
            A.RandomGridShuffle(grid=(3, 8), p=0.1),
        ], p=0.05),

    ]
    return transforms


def usual_common_transforms():
    common_transforms = [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
    ]

    rez = \
        rotate_and_scale_and_shift_transforms() \
        + noise_and_blur_transforms() \
        + common_transforms
    return rez


def get_transforms(
        add_specific_changing_color_transforms=True,
        add_usual_changing_color_transforms=True,
        add_changing_structure_transforms=True,
):
    all_transforms = []
    all_transforms += usual_common_transforms()

    if add_changing_structure_transforms:
        all_transforms += changing_structure_transforms()

    if add_specific_changing_color_transforms:
        all_transforms += specific_changing_color_transforms()

    if add_usual_changing_color_transforms:
        all_transforms += usual_changing_color_transforms()

    possible_aug = A.Compose([
        A.OneOf([
            A.Sequential(all_transforms, p=0.95),
            A.NoOp(p=0.05),
        ], p=1.),
    ])

    # todo сделать быстрее функцию  надо просто переставить их местами

    def apply_augmentation(image: np.array, resize_shape: Optional[Tuple[int, int]] = None):

        img_after_aug = possible_aug(image=image)['image']

        img_after_aug = custom_resize_fn(img_after_aug, resize_shape)

        img_after_aug = preprocess_input(image=img_after_aug)

        return img_after_aug

    return apply_augmentation


# todo разделить ауги только для SR и для LR (например rotate только для SR например
# todo исправить баг что постоянно появляются обрезанные какие-то полоски черные + на них даже
#  накладывается шум что тоже плохо

if __name__ == '__main__':

    from utils import open_image_RGB, visualize_img_from_array

    img_paths = [
        '/Users/nikita/Downloads/165253-nacionalnyj_park_banf-oblako-voda-gidroresursy-rastenie-3840x2160.jpg',
        # '/Users/nikita/Downloads/orig-3.jpeg',
        # '/Users/nikita/Downloads/lGpJ56guhdQ.jpg',
        # '/Users/nikita/Downloads/orig.png',
        # '/Users/nikita/Downloads/orig.jpeg',
        # '/Users/nikita/Downloads/0001.png',
        # '/Users/nikita/Downloads/IMG_1830.jpeg'
    ]
    transforms = get_transforms()
    for img_path in img_paths:

        for i in range(5):
            img_arr = np.array(open_image_RGB(img_path))
            after_aug = transforms(image=img_arr, )
            visualize_img_from_array(after_aug, notebook=False)

