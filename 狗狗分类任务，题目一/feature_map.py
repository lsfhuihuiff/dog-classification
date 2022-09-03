import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models

import pickle

from utils import*

class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.net = torch.load('./model/vgg16best.pkl').features.eval()

    def forward(self, x):
        out = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in [3, 8, 15, 22, 29]:
                # print(self.net[i])
                out.append(x.mean(dim=1).cpu().numpy())
        return out

transform = transforms.Compose([
    transforms.Resize([256,256]),#yuan 256
    transforms.CenterCrop([224,224]),
    transforms.ToTensor(),
])
batch_size = 4
val_imgs = ImageFolder('./dataset/val', transform=transform)
val_iter = DataLoader(val_imgs, batch_size=batch_size, shuffle=False, num_workers=0)

net = Vgg16()
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

m = 0
with torch.no_grad():
    for X,y in val_iter:
        print(X.size())
        out = net(X.to(device))
        m += 1
        if m > 0:
            break

print(len(out))


Xnp = X.cpu().numpy()
Xnp = Xnp.transpose(0, 2, 3, 1)

for i in range(4):
    plt.subplot(4,6,i*6+1)
    plt.imshow(Xnp[i])
    for j in range(5):
        layer = out[j]
        print(layer.shape)
        plt.subplot(4,6,i*6+2+j)
        plt.imshow(layer[i])


# layer3 = out[0].cpu().numpy()
# Xnp = X.cpu().numpy()
# #print(Xnp.shape())
# Xnp = Xnp.transpose(0, 2, 3, 1)
#
# for i, (map, x) in enumerate(zip(layer3,Xnp)):
#     plt.subplot(4,2, i*2+1)
#     plt.imshow(x)
#     plt.subplot(4,2, i*2+2)
#     plt.imshow(map)

plt.savefig("./tmp_image/feature_map.png")

