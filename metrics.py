import torch
import numpy as np
import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
#np.seterr(divide='ignore',invalid='ignore')


def calc_ergas(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_fus = torch.squeeze(img_fus)
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1, img_fus.shape[0])

    rmse = torch.mean((img_tgt - img_fus) ** 2, axis=1)
    rmse = rmse ** 0.5
    mean = torch.mean(img_tgt, axis=1)

    ergas = torch.mean((rmse / mean) ** 2)
    ergas = 100 / 4 * ergas ** 0.5

    return ergas.item()

def calc_psnr(img_tgt, img_fus):
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1, img_fus.shape[0])
    mse = torch.mean(torch.square(img_tgt-img_fus))
    img_max = torch.max(img_tgt)
    psnr = 10.0 * torch.log10(img_max**2/mse)

    return psnr.item()

def calc_rmse(img_tgt, img_fus):

    rmse = torch.sqrt(torch.mean((img_tgt-img_fus)**2))

    return rmse.item()

def calc_sam(img_tgt, img_fus):

    img_tgt = torch.squeeze(img_tgt)
    img_fus = torch.squeeze(img_fus)
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1, img_fus.shape[0])
    img_tgt = img_tgt / torch.max(img_tgt)
    img_fus = img_fus / torch.max(img_fus)

    A = torch.sqrt(torch.sum(img_tgt**2, axis=0))
    B = torch.sqrt(torch.sum(img_fus**2, axis=0))
    AB = torch.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)

    sam = torch.arccos(sam)
    sam = torch.mean(sam)*180/math.pi

    return sam.item()


def calc_ssim(img_tgt, img_fus):
    '''
    平均结构相似性
    :param reference:
    :param target:
    :return:
    '''
    img_tgt = img_tgt.detach().cpu().numpy()
    img_fus = img_fus.detach().cpu().numpy()
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[0])
    img_fus = img_fus.reshape(-1, img_fus.shape[0])
    ssim = structural_similarity(img_tgt, img_fus, data_range=1.0, multichannel=True)
    return ssim