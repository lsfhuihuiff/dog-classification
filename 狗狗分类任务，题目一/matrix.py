import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
n_categories = 131
train_iter, val_iter = load_data(batch_size)

net = torch.load('./model/vgg16best.pkl')
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
#device = list(net.parameters())[0].device
#print(device)
net.eval()

confusion = torch.zeros(n_categories, n_categories)
acc_sum, n = 0.0, 0

with torch.no_grad():

    for X,y in val_iter:
        pred = net(X.to(device))
        pred_index = (pred).argmax(dim=1)

        for m,k in zip(y.cpu().numpy(), pred_index.cpu().numpy()):
            confusion[m][k] += 1
        acc_sum += ((pred).argmax(dim=1)  == y.to(device)).float().sum().cpu().item()
        n += y.shape[0]

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

print(acc_sum,n)
acc = acc_sum/n

print(acc)
print('acc: %.4f'%acc)

#draw
#set up plot
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
confusion = confusion[0:10,0:10]
print(confusion.size())

cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

#set up axes
all_categories = []
with open('label.pkl', 'rb') as file:
    label = pickle.loads(file.read())
    for key, value in label.items():
        class_name = value.split('-')[2]
        all_categories.append(class_name)
    #print(all_categories)

ax.set_xticklabels([''] + all_categories[0:10], rotation=90)
ax.set_yticklabels([''] + all_categories[0:10])

#force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


plt.savefig("./tmp_image/classify_matrix.png")
plt.show()
