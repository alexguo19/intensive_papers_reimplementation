import os
import torch
import torch.nn.functional as F

def save_model(model, filename):
    """Saves the model state dictionary to the specified path."""
    filename = os.path.join("checkpoints", filename+".pth")
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    """Loads the model state dictionary from the specified path."""
    filename = os.path.join("checkpoints", filename+".pth")
    model.load_state_dict(torch.load(filename))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg}"


class AccuracyMeter(object):
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.meters = [AverageMeter("Acc@%d" % i) for i in topk]
    
    def reset(self):
        for meter in self.meters:
            meter.reset()
    
    def update(self, predict, target):
        acc = accuracy(predict, target, self.topk)
        num = target.size(0)
        for meter, val in zip(self.meters, acc):
            meter.update(val, num)

    def __str__(self):
        return " ".join(str(meter) for meter in self.meters)
    
    def as_dict(self):
        return {meter.name: meter.avg for meter in self.meters}


class Metric(object):
    """Base metric class for tracking statistics"""
    def __init__(self, func):
        super().__init__()
        self.clear()
        self.func = func
    
    def clear(self):
        self.num = 0
        self.val = 0

    def update(self, *args, **kwargs):
        n, v = self.func(*args, **kwargs)
        self.num += n
        self.val += v
    
    def average(self):
        return self.val / self.num if self.num > 0 else 0


class CrossEntropyLossMetric(Metric):
    """Metric for computing cross entropy loss"""
    def __init__(self):
        super().__init__(func=self._cross_entropy_loss_func)
    
    @staticmethod
    def _cross_entropy_loss_func(predict, target):
        n = target.size(0)
        v = F.cross_entropy(predict, target).item()
        return n, v


class AccuracyMetric(Metric):
    """Metric for computing accuracy"""
    def __init__(self):
        super().__init__(self._accuracy_func)
    
    @staticmethod
    def _accuracy_func(predict, target):
        n = target.size(0)
        p = torch.max(predict.data, dim=1)[1]
        v = (target == p).sum().item()
        return n, v
