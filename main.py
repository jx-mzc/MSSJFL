from datasets import Datasets, CaveDataset
import h5py
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
import torch
import argsParser
from train import modelTrain

args = argsParser.argsParser()
print(args)


def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'GF5':
        mat_save_path = ' '
        trainDataset = Datasets(mat_save_path)
        trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.num_workers)
        testDataset = Datasets(mat_save_path, isTrain=False)
        testLoader = DataLoader(testDataset, batch_size=1, num_workers=args.num_workers)
        modelTrain(trainLoader, testLoader, 280, 4, args.ratio, device, args.dataset)

    if args.dataset == 'cave':
        mat_save_path = ' '
        trainDataset = CaveDataset(mat_save_path)
        trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.num_workers)
        testDataset = CaveDataset(mat_save_path, isTrain=False)
        testLoader = DataLoader(testDataset, batch_size=1, num_workers=args.num_workers)
        modelTrain(trainLoader, testLoader, 31, 3, args.ratio, device, args.dataset)



if __name__ == '__main__':
    main()



