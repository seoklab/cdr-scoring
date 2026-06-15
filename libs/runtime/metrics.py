import torch


class MeanAbsoluteError:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def __call__(self, pred, target):
        pred = pred.detach()
        target = target.detach()
        value = torch.abs(pred - target)
        self._sum += float(value.sum().cpu())
        self._count += int(value.numel())

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
