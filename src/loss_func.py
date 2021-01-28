import torch 
import torch.nn as nn 
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

class LosslessTripletLoss(torch.nn.Module):
    def __init__(self, beta=None, eps=1e-8, reduction='mean'):
        #beta - The scaling factor, number of dimensions by default.
        #eps - The Epsilon value to prevent ln(0)
        super(LosslessTripletLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        assert self.reduction in ('none', 'mean', 'sum')

    def forward(self, a, p, n):
        '''
        Arguments:
            a - anchor vectors with values in range (0, 1)
            p - positive vectors with values in range (0, 1)
            n - negative vectors with values in range (0, 1)
        '''
        assert a.shape == p.shape, 'Shapes dont match.'
        assert a.shape == n.shape, 'Shapes dont match.'

        N = a.shape[1]
        beta = N if self.beta is None else self.beta
        dist_p = (a - p).pow(2).sum(dim=1)
        dist_n = (a - n).pow(2).sum(dim=1)
        dist_p = -torch.log(-(    dist_p) / beta + 1 + self.eps)
        dist_n = -torch.log(-(N - dist_n) / beta + 1 + self.eps)
        out = dist_n + dist_p
        if self.reduction == 'none':
            return out
        elif self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            raise ValueError('Unknown reduction type')