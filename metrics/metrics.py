import torch
from piq import FSIMLoss, PieAPP, DISTS, GMSDLoss, MultiScaleGMSDLoss, MDSILoss, HaarPSILoss
import torch.nn as nn
from typing import Dict


metrics_parameters = {
    FSIMLoss: [{'chromatic': False}, {'chromatic': True}],
    PieAPP: [{'stride': 27}, {'stride': 25}, {'stride': 17}, {'stride': 37}],
    DISTS: [{'mean': [0., 0., 0.], 'std': [1., 1., 1.]}],
    GMSDLoss: [{}],
    MultiScaleGMSDLoss: [{'chromatic': False}, {'chromatic': True}],
    MDSILoss: [{}],
    HaarPSILoss: [{}],
}


DEFAULT_METRICS = {}
for metric, params in metrics_parameters.items():
    for param in params:
        created_metric = metric(**param)
        created_metric.eval()
        DEFAULT_METRICS[type(created_metric).__name__+str(param)] = created_metric


class DefaultMetrics:
    def __init__(self, metrics=DEFAULT_METRICS):
        self.metrics = metrics

    def add_metrics(self, metrics: Dict[str, nn.Module]):
        self.metrics.update(metrics)

    @torch.no_grad()
    def calculate_metrics(self, pred_imgs, real_imgs):
        rez = {}
        for params, metric in self.metrics.items():
            rez[params] = metric(x=pred_imgs, y=real_imgs).item()
        return rez


if __name__ == '__main__':
    metrics = DefaultMetrics()

    x = torch.rand((3, 3, 256, 256))
    y = torch.rand((3, 3, 256, 256))
    rezs = metrics.calculate_metrics(x, y)
    for rez in rezs.items():
        print(rez)
