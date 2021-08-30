from torch import nn
import torch
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CSResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CSResBlock, self).__init__()
        self.con1 = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.con2 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.ca = ChannelAttentionModule(out_channel)
        self.sa = SpatialAttentionModule()


    def forward(self, x):
        out = self.con1(x)
        out = F.leaky_relu(out, True)
        out = self.con2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return x + out

class CResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CResBlock, self).__init__()
        self.con1 = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.con2 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.ca = ChannelAttentionModule(out_channel)

    def forward(self, x):
        out = self.con1(x)
        out = F.leaky_relu(out, True)
        out = self.con2(out)
        out = self.ca(out) * out
        return x + out

class SResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SResBlock, self).__init__()
        self.con1 = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.con2 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        out = self.con1(x)
        out = F.leaky_relu(out, True)
        out = self.con2(out)
        out = self.sa(out) * out
        return x + out

class Net(nn.Module):
    def __init__(self, hsi_channel, msi_channel, ratio, fh=256, fm=32):
        super(Net, self).__init__()
        self.ratio = ratio
        self.conv1 = nn.Sequential(
            nn.Conv2d(hsi_channel + msi_channel, fh, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.block1 = nn.Sequential(
            CSResBlock(fh, fh),
            CSResBlock(fh, fh),
            CSResBlock(fh, fh),
            CSResBlock(fh, fh),
            CSResBlock(fh, fh)
        )
        self.upscale = nn.Sequential(
            nn.Conv2d(fh, ratio * ratio * fh, kernel_size=1, stride=1),
            nn.PixelShuffle(ratio),
            nn.Conv2d(fh, hsi_channel, kernel_size=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hsi_channel + msi_channel, fm, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.block2 = nn.Sequential(
            CSResBlock(fm, fm),
            CSResBlock(fm, fm),
            CSResBlock(fm, fm),
            CSResBlock(fm, fm),
            CSResBlock(fm, fm)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fm, hsi_channel, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(hsi_channel, fh, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.block3 = nn.Sequential(
            CResBlock(fh, fh),
            CResBlock(fh, fh),
            CResBlock(fh, fh),
            CResBlock(fh, fh),
            CResBlock(fh, fh)
        )
        self.upscale2 = nn.Sequential(
            nn.Conv2d(fh, ratio * ratio * fh, kernel_size=1, stride=1),
            nn.PixelShuffle(ratio)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(msi_channel, fm, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.block4 = nn.Sequential(
            SResBlock(fm, fm),
            SResBlock(fm, fm),
            SResBlock(fm, fm),
            SResBlock(fm, fm),
            SResBlock(fm, fm)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(fh + fm, hsi_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(hsi_channel, hsi_channel, kernel_size=ratio, stride=ratio),
            nn.LeakyReLU()
        )
        self.conv_trans = nn.ConvTranspose2d(hsi_channel, hsi_channel, kernel_size=ratio, stride=ratio)
        self.conv8 = nn.Sequential(
            nn.Conv2d(hsi_channel, msi_channel, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(msi_channel, hsi_channel,  kernel_size=1, stride=1)
        )

    def forward(self, y, z):
        y_up = F.interpolate(y, scale_factor=self.ratio, mode='bilinear')
        z_down = F.interpolate(z, scale_factor=1/self.ratio, mode='bicubic')

        yz = torch.cat((y, z_down), 1)
        yz_ = self.conv1(yz)
        yz__ = self.block1(yz_)
        yz__ = yz__ + yz_
        x_yz = self.upscale(yz__)

        zy = torch.cat((z, y_up), 1)
        zy_ = self.conv2(zy)
        zy__ = self.block2(zy_)
        zy__ = zy__ + zy_
        x_zy = self.conv3(zy__)

        y_ = self.conv4(y)
        y__ = self.block3(y_)
        y__ = y__ + y_
        x_y = self.upscale2(y__)

        z__ = self.conv5(z)
        x_z = self.block4(z__)
        x_z = x_z + z__

        xyz = self.conv6(torch.cat((x_y, x_z), 1))

        x_y_z = x_yz + x_zy + xyz

        y1 = self.conv7(x_y_z)
        dy = y - y1
        dx_y = self.conv_trans(dy)

        z1 = self.conv8(x_y_z)
        dz = z - z1
        dx_z = self.conv9(dz)

        x_y_z_ = x_y_z + dx_y + dx_z

        return x_y_z_
