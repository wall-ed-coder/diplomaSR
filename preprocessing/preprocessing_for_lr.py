import albumentations as A

from preprocessing.preprocessing import SIZES_FOR_CROPS, DOWN_SCALE_COEF

# todo добавить разную интерполяцию
RESIZE_SCALE_DOWN_LR = {
    (height//scale_coef, width//scale_coef): A.Compose([
        A.Resize(height=height//scale_coef, width=width//scale_coef, always_apply=True)
    ]) for height, width in SIZES_FOR_CROPS
    for scale_coef in DOWN_SCALE_COEF
}


def noise_and_blur_transforms() -> A.Compose:
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


def scale_transforms() -> A.Compose:
    scale_transforms = [
        A.OneOf([
            A.Downscale(scale_min=0.4, scale_max=0.9, p=0.05),
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.1),
        ], p=0.05),

    ]
    return A.Compose(scale_transforms)


def resize_for_LR(image, resize_shape):
    return RESIZE_SCALE_DOWN_LR[resize_shape](image=image)['image']
