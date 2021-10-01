import torch.nn as nn


class DepthWiseConv(nn.Module):
    """
    if stride == 1
    input shape == output shape
    if kernel_size = 4, stride = 2, output shape = [H/2, W/2] where input shape = [H, W]
    """
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super(DepthWiseConv, self).__init__()
        self.depthconv = nn.Conv2d(cin, cin, kernel_size=kernel_size, stride=stride, padding=padding, groups=cin)
        self.pointconv = nn.Conv2d(cin, cout, kernel_size=1, groups=cout)

    def forward(self, x):
        out = self.depthconv(x)
        out = self.pointconv(out)
        return out


class DSConv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding=1):
        super(DSConv, self).__init__()
        self.dsconv = DepthWiseConv(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.innorm = nn.InstanceNorm2d(cout)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.convB = ConvBlock(cout, cout, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.dsconv(x)
        out = self.innorm(out)
        out = self.lrelu(out)
        out = self.convB(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, activate=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.innorm = nn.InstanceNorm2d(cout)
        self.activation = nn.LeakyReLU(inplace=True)
        self.activate = activate

    def forward(self, x):
        out = self.conv(x)
        out = self.innorm(out)
        if self.activate:
            out = self.activation(out)
            return out
        else:
            return out


class InvertedResidualBlock(nn.Module):
    """
    input shape == output shape
    input channel: 256
    output channel: 256
    """
    def __init__(self,cin, cout):
        super(InvertedResidualBlock, self).__init__()
        assert (cin == cout)
        self.convB = ConvBlock(cin, cin*2, kernel_size=1, stride=1)
        self.dpconv = DepthWiseConv(cin*2, cin*2, kernel_size=3, stride=1, padding=1)
        self.innorm = nn.InstanceNorm2d(cin*2)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(cin*2, cout, kernel_size=1, stride=1)
        self.innorm2 = nn.InstanceNorm2d(cout)

    def forward(self, x):
        y = self.convB(x)
        y = self.dpconv(y)
        y = self.innorm(y)
        y = self.lrelu(y)
        y = self.conv(y)
        y = self.innorm2(y)
        out = y + x
        return out


class DownSample(nn.Module):
    def __init__(self, cin, cout):
        super(DownSample, self).__init__()
        #  input shape: [H, W]; output shape: [H/2, W/2]
        # down conv, shape be [H/2, W/2]
        self.dsconv = DSConv(cin, cin, kernel_size=4, stride=2, padding=1)
        # x_feature go in
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.dsconv2 = DSConv(cin, cout, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # y1 shape: [H/2, W/2]
        y1 = self.dsconv(x)
        # y2 shape: [H/2, W/2]
        y2 = self.downsample(x)
        y2 = self.dsconv2(y2)
        out = y1 + y2
        return out


class UpSample(nn.Module):
    def __init__(self, cin, cout):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dsconv = DSConv(cin, cout, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.dsconv(out)
        return out

