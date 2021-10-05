from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import tqdm


class PreprocessDataSet(Dataset):

    def __init__(self, style_path, content_path, size=256):
        super(PreprocessDataSet, self).__init__()
        self.style_path = style_path
        self.content_path = content_path
        self.s_len, self.c_len = len(style_path), len(content_path)
        self.l = min(self.s_len, self.c_len)
        if size > 256:
            rsizer = transforms.Resize(size, interpolation=InterpolationMode.BICUBIC)
        else:
            rsizer = transforms.RandomCrop(size)
        tran = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            rsizer,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        ])

        self.content_images, self.style_images = self.datalist(self.l, tran)

    def datalist(self, data_len, transform):
        np.random.shuffle(self.content_path)
        np.random.shuffle(self.style_path)
        content_images = []
        style_images = []
        print("Init dataList")
        for i in tqdm.trange(data_len):
            k = self.c_len % (data_len + i)
            cp, sp = Image.open(self.content_path[k]), Image.open(self.style_path[i])
            content_images.append(transform(cp))
            style_images.append(transform(sp))
        return content_images, style_images

    def __getitem__(self, index) -> T_co:
        return zip(self.content_images[index], self.style_images[index])

    def __len__(self):
        return self.l
