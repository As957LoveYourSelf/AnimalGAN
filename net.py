import torch
import torch.nn as nn
import torch.functional as F
import blocks
from functions import *


class Generator(nn.Module):
    def __init__(self, res_n):
        super(Generator, self).__init__()
        # add residual block
        residual_block = []
        for i in range(res_n):
            residual_block.append(blocks.InvertedResidualBlock(256, 256))

        self.generator = nn.Sequential(
            blocks.ConvBlock(3, 64, kernel_size=3, stride=1, padding=1),
            blocks.ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            blocks.DownSample(64, 128),
            blocks.ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            blocks.DSConv(128, 256, kernel_size=3, stride=1),
            blocks.DownSample(256, 256),
            blocks.ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            *residual_block,
            blocks.ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            blocks.UpSample(256, 128),
            blocks.DSConv(128, 128, kernel_size=3, stride=1),
            blocks.ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            blocks.UpSample(64, 64),
            blocks.ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            blocks.ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def generate(self, input_features, device='gpu'):
        input_features.to(device)
        out_features = self.generator(input_features)
        return out_features

    def forward(self, input_features, content_feature, grayscale, wadv=300, wcon=1.5, wgra=3, wcol=50):
        """
        loss = Wadv*(pow2(G(p) - 1)) + Wcon*Lcon(G, D) + WgraLgra(G, D) + WcolLcol(G, D)
        """
        g_image = self.generate(input_features)
        g_content = self.generate(content_feature)
        g_loss = wadv * torch.square(input_features - 1)
        t_loss = wcon * content_loss(g_image, content_feature) + wgra * gram_loss(grayscale,
                                                                                  g_image) + wcol * color_loss(
            content_feature, g_content)
        return g_loss + t_loss


class Discriminator(nn.Module):
    def __init__(self, input_feature):
        """
        input shape: [batch, channel, H, W]
        H = 256, W = 256
        """
        super(Discriminator, self).__init__()
        self.discriminate = nn.Sequential(
            # shape = [256, 256]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            # shape = [128, 128]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # shape = [64, 64]
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
        )

    def discriminator(self, x, device='gpu'):
        x.to(device)
        out = self.discriminate(x)
        return out

    def forward(self, generate_content_feature, style_feature, gray_style_feature, smooth_gray_feature):
        d_loss = torch.squeeze(self.discriminator(style_feature) - 1.0)
        t_loss = torch.squeeze(self.discriminator(generate_content_feature)) + torch.squeeze(
            self.discriminator(gray_style_feature)) + 0.1*torch.squeeze(self.discriminator(smooth_gray_feature))
        return d_loss + t_loss


class AnimalGAN(nn.Module):
    def __init__(self):
        super(AnimalGAN, self).__init__()
        pass
