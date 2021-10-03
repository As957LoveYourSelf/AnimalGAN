import torch
import torch.nn.functional as F
import numpy as np


def rgb2yuv(rgb_tensor):
    _rgb_to_yuv_kernel = [[0.299, -0.14714119, 0.61497538],
                          [0.587, -0.28886916, -0.51496512],
                          [0.114, 0.43601035, -0.10001026]]
    # _rgb_to_yuv_kernel = np.array(_rgb_to_yuv_kernel, dtype=np.float32)
    # _rgb_to_yuv_bias = np.array([0., 0.5, 0.5], dtype=np.float32)
    #
    # _rgb_to_yuv_kernel = torch.from_numpy(_rgb_to_yuv_kernel)
    # _rgb_to_yuv_bias = torch.from_numpy(_rgb_to_yuv_bias)
    #
    # temp = F.conv2d(rgb_tensor, _rgb_to_yuv_kernel, _rgb_to_yuv_bias)
    # return temp

    assert (isinstance(rgb_tensor, torch.Tensor))
    assert (len(rgb_tensor.shape == 4))
    b, c, h, w = rgb_tensor.shape
    rgb_tensor = rgb_tensor.view(b, c, h*w).transpose(1, 2)
    _rgb_to_yuv_kernel = np.array(_rgb_to_yuv_kernel, dtype=np.float32)
    _rgb_to_yuv_kernel = torch.from_numpy(_rgb_to_yuv_kernel)
    temp = torch.tensordot(rgb_tensor, _rgb_to_yuv_kernel, dims=1).transpose(1, 2)
    return temp.view(b, c, h, w)


def get_yuv(yuv_tensor):
    """
    input: YUV Tensor, shape = [batch, channel, H, W]
    return: Y, U, V
    """
    y = yuv_tensor[:, 0]
    u = yuv_tensor[:, 1]
    v = yuv_tensor[:, 2]
    return y, u, v








