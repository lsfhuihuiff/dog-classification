import os
import sys
import numpy as np
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

batch_size =64
#train_iter, val_iter = load_data(batch_size)

transform1 = transforms.Compose([
    transforms.Resize([424,424]),#yuan 256
    transforms.CenterCrop([386,386]),
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.Resize([256,256]),#yuan 256
    transforms.CenterCrop([224,224]),
    transforms.ToTensor(),
])
val_imgs1 = ImageFolder('./dataset/val', transform=transform1)
val_imgs2 = ImageFolder('./dataset/val', transform=transform2)

val_iter1 = DataLoader(val_imgs1, batch_size=batch_size, shuffle=False, num_workers=0)
val_iter2 = DataLoader(val_imgs2, batch_size=batch_size, shuffle=False, num_workers=0)

net = torch.load('./model/vgg16best.pkl')
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()
acc_sum, n = 0.0, 0
preds1 = []

with torch.no_grad():
    for X,y in val_iter1:
        pred1 = net(X.to(device))
        #acc_sum += (pred).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        preds1.append(pred1)
    m=0
    for X,y in val_iter2:
        pred2 = net(X.to(device))
        pred = 0.5*preds1[m] + 0.5* pred2
        acc_sum += ((pred).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        n += y.shape[0]
        m += 1

acc = acc_sum/n
print(acc)
print('acc: %.4f'%acc)
