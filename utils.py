import torch
import cv2
import numpy as np
import os
import glob
import tqdm


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


def image_to_gray(image_dat):
    return cv2.cvtColor(image_dat, cv2.COLOR_BGR2GRAY)


def images2gray(images_path, save_path):
    images = glob.glob(os.path.join(images_path, "*.jpg"))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img in tqdm.tqdm(images):
        filename = os.path.basename(img)
        image = cv2.imread(img)
        image = image_to_gray(image)
        cv2.imwrite(os.path.join(save_path, "gray_"+filename), image)
    print("Done")


if __name__ == '__main__':
    imagedata_path = "./cartoonDataset"
    style_paths = ["Hayao", "Paprika", "Shinkai", "SummerWar"]
    type_paths = ["smooth", "style"]
    for s in style_paths:
        f_path = os.path.join(imagedata_path, s)
        for t in type_paths:
            tf_path = os.path.join(f_path, t)
            images2gray(tf_path, os.path.join(f_path, "gray_"+t))













