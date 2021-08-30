'''数据集类'''
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio
from utils import normalize


class Datasets(Dataset):
    def __init__(self, mat_save_path, patch_size=12, stride=6, ratio=3, isTrain=True):
        super(Datasets, self).__init__()
        self.mat_save_path = mat_save_path
        self.stride = stride
        self.rows = 60
        self.cols = 60
        # 生成样本和标签
        if isTrain:
            self.hsi_data, self.msi_data, self.label = self.generateTrain(patch_size, ratio, num_star=1, num_end=21, s=9)
        else:
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_size=60, ratio=3, num_star=21, num_end=29, s=1)
            #self.hsi_data, self.msi_data, self.label = self.generateTest(patch_size=60, ratio=3, num_star=29, num_end=37, s=1)

    def generateTrain(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 280), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 280), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']
            # 数据类型转换
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
            for x in range(0, self.rows - patch_size + self.stride, self.stride):
                for y in range(0, self.cols - patch_size + self.stride, self.stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio :(x + patch_size) * ratio,y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 280), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 280), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']
            # 数据类型转换
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
            for x in range(0, self.rows - patch_size + patch_size, patch_size):
                for y in range(0, self.cols - patch_size + patch_size, patch_size):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        return torch.tensor(self.hsi_data[index], dtype=torch.float32), torch.tensor(self.msi_data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32)

    def __len__(self):
        return self.label.shape[0]

class CaveDataset(Dataset):
    def __init__(self, mat_save_path, patch_size=10, stride=6, ratio=8, isTrain=True):
        super(CaveDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.stride = stride
        self.rows = 64
        self.cols = 64
        # 生成样本和标签
        if isTrain:
            self.hsi_data, self.msi_data, self.label = self.generateTrain(patch_size, ratio, num_star=1, num_end=21, s=10)
        else:
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_size=64, ratio=ratio, num_star=21, num_end=27, s=1)
            #self.hsi_data, self.msi_data, self.label = self.generateTest(patch_size, ratio, num_star=27, num_end=33, s=1)


    def generateTrain(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']
            # 数据类型转换
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
            for x in range(0, self.rows - patch_size + self.stride, self.stride):
                for y in range(0, self.cols - patch_size + self.stride, self.stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']
            # 数据类型转换
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
            for x in range(0, self.rows - patch_size + patch_size, patch_size):
                for y in range(0, self.cols - patch_size + patch_size, patch_size):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        return torch.tensor(self.hsi_data[index], dtype=torch.float32), torch.tensor(self.msi_data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.float32)

    def __len__(self):
        return self.label.shape[0]
