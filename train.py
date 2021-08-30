import torch
from model.models import Net
from utils import weight_init, train, validate
from torch import nn, optim


def modelTrain(trainLoader, testLoader, hsi_channel, msi_channel, ratio, device, dataset):
    SEED = 971226
    MODEL_NAME = dataset
    torch.manual_seed(SEED)
    model = Net(hsi_channel, msi_channel, ratio).to(device)
    # 初始化模型参数
    model.apply(weight_init)
    # 损失函数，优化器，学习率下架管理器
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 500, 2000], gamma=0.1, last_epoch=-1)
    for epoch in range(10000):
        model, train_loss = train(model, trainLoader, optimizer, criterion, device)
        eval_loss, psnr = validate(model, testLoader, criterion, device)
        scheduler.step()
        print('Epoch: {}  Train Loss: {:.6f}   Eval Loss:{:.6f}   psnr:{:.6f}'.format(epoch + 1, train_loss, eval_loss, psnr))

        if epoch == 0:
            torch.save(model.state_dict(), './data/model/{}/{}_epoch_{}.pkl'.format(MODEL_NAME, MODEL_NAME, epoch+1))
            psnr_max = psnr
            best_epoch = epoch + 1

        if psnr > psnr_max:
            torch.save(model.state_dict(),'./data/model/{}/{}_best.pkl'.format(MODEL_NAME, MODEL_NAME))
            psnr_max = psnr
            best_epoch = epoch + 1
            with open('./data/{}_best_psnr.txt'.format(MODEL_NAME), 'a') as f:
                f.write('{}_best_epoch:'.format(MODEL_NAME) + str(best_epoch) + ',   psnr:' + str(psnr) + ',  train loss:' + str(train_loss) + ',  eval loss:' + str(eval_loss) + '\n')







