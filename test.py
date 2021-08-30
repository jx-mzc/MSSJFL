import torch
from model.models import Net
from utils import weight_init
from torch import nn, optim
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_ssim
import numpy as np
import h5py
import scipy.io as scio
from datasets import Datasets, CaveDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # mat_save_path = './data/GF5_S2A/'
    # testDataset = CaveDataset(mat_save_path, isTrain=False)
    # testLoader = DataLoader(testDataset, batch_size=1)

    mat_save_path = './data/CAVEMAT/'
    testDataset = Datasets(mat_save_path, isTrain=False)
    testLoader = DataLoader(testDataset, batch_size=1)

    model_path = './data/model/cave/cave_best.pkl'
    model = Net(hsi_channel=31, msi_channel=3, ratio=8).to(device)
    model.load_state_dict(torch.load(model_path))


    psnr = []
    rmse = []
    ergas = []
    sam = []
    ssim = []
    model.eval()

    i = 0
    for hsi_input, msi_input, label in tqdm(testLoader):
        hsi_inputs, msi_inputs, labels = hsi_input.to(device), msi_input.to(device), label.to(device)
        hsi_inputs, msi_inputs, labels = hsi_inputs.permute((0, 3, 1, 2)), msi_inputs.permute((0, 3, 1, 2)), labels.permute(
            (0, 3, 1, 2))
        with torch.no_grad():
            out = model(hsi_inputs, msi_inputs)

        pred = np.transpose(out.cpu().numpy(), (0, 2, 3, 1))
        scio.savemat('./data/img/cave/img{}.mat'.format(i+1),{'lrhsi': hsi_input.cpu().numpy()[0], 'hrmsi': msi_input.cpu().numpy()[0],'hrhsi': label.cpu().numpy()[0], 'fusion': pred[0]})
        i += 1
        psnr.append(calc_psnr(labels, out))
        rmse.append(calc_rmse(labels, out))
        ergas.append(calc_ergas(labels, out))
        sam.append(calc_sam(labels, out))
        ssim.append(calc_ssim(labels, out))
        print('Psnr: {:.6f}'.format(float(np.mean(psnr))))
        print('Rmse:   {:.6f}'.format(float(np.mean(rmse))))
        print('Ergas:   {:.6f}'.format(float(np.mean(ergas))))
        print('Sam:   {:.6f}'.format(float(np.mean(sam))))
        print('Ssim:   {:.6f}'.format(float(np.mean(ssim))))

if __name__ == '__main__':
    test()

