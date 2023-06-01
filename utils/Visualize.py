import visdom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

LABEL_NAMES = ['person', 'bicycle', 'car', 'bus', 'motorbike']

class Visualize(object):
    def __init__(self, env='main'):
        self.vis = visdom.Visdom(env=env)

        self.iter = {}

    def plot(self, name, y, **kwargs):
        x = self.iter.get(name, 0)
        self.vis.line(Y=[y.item()], X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.iter[name] = x + 1

    def imshow(self, name, img, **kwargs):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))
        self.vis.images(img,
                        win=name,
                        opts=dict(title=name),
                        **kwargs)

    def visdom_bbox(self, *args, **kwargs):
        fig = vis_bbox(*args, **kwargs)
        data = fig4vis(fig)
        return data

    def txt(self, text, win):
        self.vis.text(text, win=win)

def vis_img(img=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        img = img.transpose((1, 2, 0)).astype(np.uint8)
        ax.imshow(img)
    return ax

def vis_bbox(img, bboxes, labels=None, scores=None, ax=None):

    ax = vis_img(img, ax=ax)
    if ax.get_title() == '':
        ax.set_title(len(bboxes))

    if len(bboxes) == 0:
        return ax

    palette = sns.color_palette('muted', len(LABEL_NAMES))

    for i, bbox in enumerate(bboxes):
        xy = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        lb = 0 if labels is None else labels[i]
        ax.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=palette[lb], linewidth=2))

        if labels is not None and img is not None:
            label_txt = LABEL_NAMES[labels[i]]
            if scores is not None and i < len(scores):
                label_txt += ': ' + '{:.2f}'.format(scores[i])

            ax.text(bbox[0], bbox[1], label_txt,
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax



def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it

    @param fig: a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plt.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.