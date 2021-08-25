import albumentations as A


def save_transform(path: str, transform: A.Compose):
    A.save(transform, path)


def load_transform(path: str, dict_for_load_lambdas: dict) -> A.Compose:
    return A.load(path, lambda_transforms=dict_for_load_lambdas)
