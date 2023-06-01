import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import torch
from torch.nn.functional import conv2d
from torchvision.transforms import Grayscale

def rgb2gray(img):
    im = Image.fromarray(img)
    im = im.convert('L')
    gray_img = np.array(im)
    return gray_img / 255.


def guided_filter(img, raw_t, r=40, eps=1e-3):
    I = rgb2gray(np.transpose(img, (1, 2, 0)))
    mean_I = uniform_filter(I, size=r)
    mean_t = uniform_filter(raw_t, size=r)
    mean_It = uniform_filter(I * raw_t, size=r)

    cov_It = mean_It - mean_I * mean_t

    mean_II = uniform_filter(I * I, size=r)
    var_I = mean_II - mean_I * mean_I

    a = cov_It / (var_I + eps)
    b = mean_t - a * mean_I

    mean_a = uniform_filter(a, size=r)
    mean_b = uniform_filter(b, size=r)

    q = mean_a * I + mean_b
    return q

def guided_filter_tensor(img, raw_t, r=40, eps=1e-3):
    I = Grayscale(img)

    box_filter = torch.ones((1, r, r))

    mean_I = conv2d(I, box_filter)
    print(mean_I.shape)
