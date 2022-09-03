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
import copy

import matplotlib.pyplot as plt


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomChoice([transforms.RandomResizedCrop(size=224),
                                 transforms.RandomResizedCrop(size=386)]),
        #transforms.RandomResizedCrop(size=224),
        #transforms.RandomHorizontalFlip(),
        #transforms.Resize([224,224]),#yuan 256
        #transforms.CenterCrop([224,224]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 406], [0.229, 0.224, 0.225])
    ])
    transform1 = transforms.Compose([
        # transforms.RandomChoice([transforms.RandomResizedCrop(size=224),
        #                          transforms.RandomResizedCrop(size=386)]),
        transforms.RandomResizedCrop(size=224),
        #transforms.RandomErasing(p=0.8, scale=(0.02, 0.1), ratio=(0.5,2), value=0),
        transforms.RandomHorizontalFlip(),
        #transforms.Resize([224,224]),#yuan 256
        #transforms.CenterCrop([224,224]),
        transforms.ToTensor(),
       # transforms.Normalize([0.485, 0.456, 406], [0.229, 0.224, 0.225])
    ])

    transform2 = transforms.Compose([
        #transforms.RandomResizedCrop(size=224),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize([256,256]),#yuan 256
        transforms.CenterCrop([224,224]),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 406], [0.229, 0.224, 0.225])
    ])

    # train_imgs = ImageFolder('./dataset-less/train', transform=transform1)
    # val_imgs = ImageFolder('./dataset-less/val', transform=transform2)

    train_imgs = ImageFolder('./dataset/train', transform=transform1)
    val_imgs = ImageFolder('./dataset/val', transform=transform2)

    train_iter = DataLoader(train_imgs, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iter = DataLoader(val_imgs, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, val_iter

def train(net, train_iter, val_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print('training on', device)

    train_acc = []
    val_acc = []
    best_acc = 0.0

    # trans = transforms.Compose([transforms.Resize([386, 386])])

    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        loss_sum, acc_sum, v_acc, n, batch_count, start = 0.0, 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            # if np.random.rand(1) < 0.5:
            #     X = trans(X)

            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss_sum += l.cpu().item()
            acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            print(batch_count)

        v_acc = evaluate_accuracy(val_iter, net)
        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(net.state_dict())

        train_acc.append(acc_sum/n)
        val_acc.append(v_acc)

        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f seconds'
              %(epoch + 1, loss_sum/batch_count, acc_sum/n, v_acc, time.time()-start))

        # if epoch%50 == 0:
        #     train_iter, val_iter = load_data(batch_size)
        # if epoch%30 == 0:
        #     torch.save(net.load_state_dict(best_wts), './vgg16-'+str(epoch)+'.pkl')
        print('best_acc:', best_acc)
    return train_acc, val_acc, best_acc, best_wts

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0

    with torch.no_grad():
        for X,y in data_iter:

            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code___.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()

            n += y.shape[0]

    return acc_sum/n

def draw_picture(train_acc, val_acc):
    x = np.arange(1, len(train_acc)+1, 1)
    y1 = np.array(train_acc)
    y2 = np.array(val_acc)
    plt.plot(x, y1, label='train')
    plt.plot(x, y2, linestyle='--', label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('train&test')
    plt.legend()
    plt.savefig('./train&test.png')
    plt.show()
