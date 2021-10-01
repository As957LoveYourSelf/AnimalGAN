import torch
import torch.nn as nn
import torch.functional as F
import blocks


class Generator(nn.Module):
    def __init__(self, res_n):
        super(Generator, self).__init__()
        # add residual block
        residual_block = []
        for i in range(res_n):
            residual_block.append(blocks.InvertedResidualBlock(256, 256))

        self.generate = nn.Sequential(
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

    def generate_image(self, input_features):
        out_features = self.generate(input_features)
        return out_features

    def forward(self, input_features, wadv=300, wcon=1.5, wgra=3, wcol=50):
        pass


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
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self):
        pass


class AnimalGAN(nn.Module):
    def __init__(self):
        super(AnimalGAN, self).__init__()
        pass
























