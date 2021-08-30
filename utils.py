from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import init
from metrics import calc_psnr, calc_ssim, calc_ergas, calc_rmse, calc_sam
import cv2
import os
from tqdm import tqdm


def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)

# 数据归一化
def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def train(model, trainLoader, optimizer, criterion, device):
    train_loss = []
    model.train()
    for step, (hsi_input, msi_input, label) in enumerate(trainLoader):
        hsi_input, msi_input, label = hsi_input.to(device), msi_input.to(device), label.to(device)
        hsi_input, msi_input, label = hsi_input.permute((0, 3, 1, 2)), msi_input.permute((0, 3, 1, 2)),  label.permute((0, 3, 1, 2))

        #forward
        pre_x = model(hsi_input, msi_input)
        loss = criterion(pre_x, label)
        loss.requires_grad_(True)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return model, float(np.mean(train_loss))


def validate(model, testLoader, criterion, device):
    eval_loss = []
    psnr = []
    model.eval()

    for hsi_input, msi_input, label in testLoader:
        hsi_input, msi_input, label = hsi_input.to(device), msi_input.to(device), label.to(device)
        hsi_input, msi_input, label = hsi_input.permute((0, 3, 1, 2)), msi_input.permute((0, 3, 1, 2)), label.permute((0, 3, 1, 2))

        with torch.no_grad():
                pre_x = model(hsi_input, msi_input)

        loss = criterion(pre_x, label)

        eval_loss.append(loss.item())
        psnr.append(calc_psnr(label, pre_x))

    return float(np.mean(eval_loss)), float(np.mean(psnr))


