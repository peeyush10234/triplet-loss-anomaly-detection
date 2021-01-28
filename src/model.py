import torch
import torchvision.transforms as transforms
import torchvision
import random
from torch import nn

class CnnAutoEncoder(nn.Module):
  def __init__(self):
    super(CnnAutoEncoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3,4,kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(True),
        nn.Conv2d(4,8,kernel_size=5),
        nn.MaxPool2d(kernel_size=2),
        nn.Sigmoid(),
    )

    self.decoder = nn.Sequential(
       nn.Upsample(scale_factor=2),
       nn.ConvTranspose2d(8, 4 , stride=1, kernel_size=5),
       nn.ReLU(True),
       nn.Upsample(scale_factor=2),
       nn.ConvTranspose2d(4,3, stride=1, kernel_size=5),
       nn.ReLU(True)
    )

  def forward(self, x):
    x_e = self.encoder(x)
    x_d = self.decoder(x_e)
    x_e = torch.reshape(x_e, (x_e.shape[0], x_e.shape[1]*x_e.shape[2]*x_e.shape[3]))
    return x_e, x_d