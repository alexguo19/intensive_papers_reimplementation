import torch.nn as nn

def weightedCrossEntropyLoss(x,y,weights):
    l1 = (weights[0]*nn.CrossEntropyLoss()(x[0],y))
    l2 = (weights[1]*nn.CrossEntropyLoss()(x[1],y))
    l3 = (weights[2]*nn.CrossEntropyLoss()(x[2],y))
    return 0.3 * l1 + 0.3 * l2 + l3
