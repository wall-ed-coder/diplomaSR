import albumentations as A

from preprocessing.preprocessing import SHIFT_LIMIT


def rotate_and_shift_transforms() -> A.Compose:
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


def usual_changing_color_transforms() -> A.Compose:
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.1),
    ]
    return A.Compose(transforms)


def specific_changing_color_transforms() -> A.Compose:
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

