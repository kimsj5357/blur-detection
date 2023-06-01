import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
import torch

from guided_filter import guided_filter


def visualize(img, title='', cmap=None):
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

class DeHaze:
    def __init__(self,
                 tmin=0.1, win_size=15, A_rate=0.0001, Amax=255,
                 omega=0.95, guided=True, radius=100, eps=1e-3,
                 visualize=False):

        """
        :param tmin:        threshold of transmission rate
        :param win_size:    window size of the dark channel prior
        :param A_rate:      percentage of pixels for estimating the atmosphere light
        :param omega:       bias for the transmission estimate
        :param guided:      whether to use the guided filter to fine the image
        :param radius:      radius of guided filter
        :param eps:         epsilon for the guided filter
        """

        self.tmin = tmin
        self.win_size = win_size
        self.A_rate = A_rate
        self.Amax = Amax
        self.omega = omega
        self.guided = guided
        self.radius = radius
        self.eps = eps
        self.visualize = visualize

    def __call__(self, img):
        """
        :param img: input image (height, width, 3)
        :return: (dark, rawt, refinedt, rawrad, rerad)
            - dark: images for dark channel prior
            - rawt: raw transmission estimate
            - refinedt: refined transmission estimate
            - rawad: recovered radiance with raw t
            - rerad: recovered radiance with refined t
        """

        # print(img.shape)

        dark_ch, A, raw_t, refined_t = self.dehaze_raw(img)
        A = A[:, None, None]

        refined_t[refined_t < self.tmin] = self.tmin
        refined_t = np.broadcast_to(refined_t[None, ...], img.shape)

        dehaze_img = (img.astype(np.float32) - A) / refined_t + A
        # for ch in range(3):
        #     print('[%.2f, %.2f]' % (dehaze_img[:, :, ch].min(), dehaze_img[:, :, ch].max()))
        dehaze_img = np.maximum(np.minimum(dehaze_img, 255), 0).astype(np.uint8)

        if self.visualize:
            visualize(dehaze_img, 'Dehaze image')

        dark_ch = dark_ch / 255.

        return dark_ch, A, raw_t, refined_t


    def dehaze_raw(self, img):
        dark_ch = self.get_dark_channel(img)

        A = self.get_atmosphere(img, dark_ch)
        # print('atmosphere:', A)

        raw_t = self.get_transmission(img, A[:, None, None])
        # print('raw transmission: between [%.4f, %.4f]' % (raw_t.min(), raw_t.max()))

        refined_t = self.refine_transmission(img, raw_t)
        # print('refined transmission: between [%.4f, %.4f]' % (refined_t.min(), refined_t.max()))


        if self.visualize:
            visualize(img)
            visualize(dark_ch, 'dark channel', 'gray')
            visualize(raw_t, 'raw transmission', 'gray')
            visualize(refined_t, 'refined transmission', 'gray')

        return dark_ch, A, raw_t, refined_t

    def get_dark_channel(self, img):
        dark_ch = minimum_filter(img, size=self.win_size)
        dark_ch = np.min(dark_ch, axis=0)

        return dark_ch

    def get_atmosphere(self, img, dark_ch):
        h, w = dark_ch.shape
        num_A = int(h * w * self.A_rate)

        flat_img = img.reshape(3, -1)
        flat_dark = dark_ch.ravel()
        search_idx = (-flat_dark).argsort()[:num_A]
        # print('atmosphere light region:', [(i // w, i % w) for i in search_idx])
        A = np.max(flat_img[:, search_idx], axis=1)
        # A = np.minimum(A, self.Amax)
        A = np.maximum(A, 1)

        return A

    def get_transmission(self, img, A):
        return 1 - self.omega * self.get_dark_channel(img / A)


    def refine_transmission(self, img, raw_t):
        refined_t = guided_filter(img, raw_t, r=self.radius, eps=self.eps)
        return refined_t

    def get_dehaze_tensor(self, img, A, refined_t):
        t = refined_t.clone()
        t = torch.clamp(t, min=self.tmin)
        A = torch.clamp(A, min=1, max=self.Amax)

        dehaze_img = (img - A) / t + A
        dehaze_img = torch.clamp(dehaze_img, min=0, max=255)

        dehaze_img = dehaze_img / 255.

        return dehaze_img

    def _get_dehaze(self, img, A):

        raw_t = self.get_transmission(img, A)
        refined_t = self.refine_transmission(A, raw_t)

        dehaze_img = self.get_dehaze_tensor(img, A, refined_t)

        return dehaze_img