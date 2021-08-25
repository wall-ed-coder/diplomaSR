from abc import ABC
from torch import Tensor


class ABCLoss(ABC):

    def get_loss(self, pred_img, real_img) -> Tensor:
        raise NotImplementedError

