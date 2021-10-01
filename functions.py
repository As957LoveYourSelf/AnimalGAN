import torch.nn.functional as F
import torch
from torchvision.models import vgg19

vggpath = "../../pretrainmodels/vgg19-dcbb9e9d.pth"
vgg = vgg19().cuda()
pre = torch.load(vggpath)
vgg.load_state_dict(pre)
vgg = vgg[14:21]


def gram_matrix():
    pass


def adversarial_loss(generator, discriminator):
    pass


def content_loss():
    pass


def color_loss():
    pass



























