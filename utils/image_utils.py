import cv2
from PIL import Image
from IPython.display import display
import torch
import numpy as np
from typing import Union


def open_image_RGB(path_to_open: str):
    image = cv2.imread(path_to_open)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def visualize_img(img_path: str, notebook=True):
    img = Image.open(img_path)
    if notebook:
        display(img)
    else:
        img.show()


def visualize_img_from_array(img_array: Union[np.array, torch.Tensor],  notebook=True, transpose=True):
    # todo сделать чтобы notebook автоматически определялся
    if isinstance(img_array, torch.Tensor):
        img_array = img_array.cpu().numpy()

    if transpose:
        img_array = img_array.transpose((1, 2, 0))

    if img_array.max() <= 1.:
        img_array = (img_array*255).astype(np.uint8)

    img = Image.fromarray(img_array)

    if notebook:
        display(img)
    else:
        img.show()
