import torch.nn as nn
import torch
from torch import Tensor

# todo
# можно подумать на счет того какие веса чекают какие фичи например везде есть условные линии
# поэтому там порог может быть маленький


def prrelu(input: Tensor, min_range_for_line, max_range_for_line, smoothing):
    res = input.clone()

    return prrelu_(res, min_range_for_line, max_range_for_line, smoothing)


def prrelu_(input: Tensor, min_range_for_line: float, max_range_for_line: float, smoothing: float):
    return torch.where(
        input < min_range_for_line, (input - min_range_for_line) * smoothing + min_range_for_line,
        torch.where(
            max_range_for_line < input, (input - max_range_for_line) * smoothing + max_range_for_line,
            input
        )
    )


class PRReLU(nn.Module):
    """
    parametrise random relu
    will be get
    random smoothing in range(lower_smoothing, upper_smoothing)
    random window_for_line in range(min_diff_for_line, max_diff_for_line)
    random range with just usual y=x function in range(lower_x_for_line, upper_x_for_line-window_for_line)


    """

    def __init__(
            self,
            lower_smoothing: float = -0.02,
            upper_smoothing: float = 0.3,
            sample_method_for_smoothing: str = "normal",
            lower_x_for_line: float = -3.5,
            upper_x_for_line: float = 3.5,
            min_diff_for_line: float = 0.5,
            max_diff_for_line: float = 3.,
            inplace:bool = False,
    ):
        super().__init__()
        assert -1. <= lower_smoothing < 1.
        assert -1. < upper_smoothing < 1
        assert lower_smoothing < upper_smoothing
        assert upper_x_for_line - lower_x_for_line > max_diff_for_line
        assert max_diff_for_line > min_diff_for_line
        self.inplace = inplace
        if sample_method_for_smoothing == "normal":
            mean_smoothing = torch.tensor(upper_smoothing + lower_smoothing) / 2
            std_smoothing = torch.tensor(upper_smoothing - lower_smoothing) / 6
            self.smoothing = torch.normal(
                mean=mean_smoothing,
                std=std_smoothing
            ).item()
            self.smoothing = self.smoothing if not (-0.003 < self.smoothing < 0.003) else 0.005
        else:
            raise NotImplementedError()

        self.window_for_line = torch.normal(
            mean=torch.tensor(min_diff_for_line + max_diff_for_line) / 2,
            std=torch.tensor(max_diff_for_line - min_diff_for_line) / 6
        ).item()

        self.min_for_line = torch.normal(
            mean=torch.tensor(lower_x_for_line + (upper_x_for_line - self.window_for_line)) / 2,
            std=torch.tensor((upper_x_for_line - self.window_for_line) - lower_x_for_line) / 6
        ).item()
        self.min_for_line = min(
            max(self.min_for_line, lower_x_for_line), upper_x_for_line - self.window_for_line
        )

        self.max_for_line = self.min_for_line + self.window_for_line

    def forward(self, input: Tensor):
        if self.inplace:
            return prrelu_(
                input, max_range_for_line=self.max_for_line,
                min_range_for_line=self.min_for_line, smoothing=self.smoothing
            )
        else:
            return prrelu(
                input, max_range_for_line=self.max_for_line,
                min_range_for_line=self.min_for_line, smoothing=self.smoothing
            )

    def extra_repr(self):
        return 'smoothing={}, window_for_line={}, min_for_line={}, max_for_line={}' \
            .format(self.smoothing, self.window_for_line, self.min_for_line, self.max_for_line)
