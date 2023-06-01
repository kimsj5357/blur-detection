import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from torch.utils.data import Dataset

def inverse_normalize(img):
    return img * 255.

class DehazeDataset(Dataset):
    def __init__(self, root_dir, size=512):
        self.root_dir = root_dir
        self.img_path = os.path.join(root_dir, 'JPEGImages')
        self.anno_path = os.path.join(root_dir, 'Annotations')

        self.ids = os.listdir(self.img_path)
        self.label_names = ['person', 'bicycle', 'car', 'bus', 'motorbike']
        self.img_size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx].split('.')[0]
        anno = ET.parse(os.path.join(self.anno_path, img_name + '.xml'))

        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                float(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        im = Image.open(os.path.join(self.img_path, img_name + '.png'))

        # img = np.array(img)
        # scale = [1., 1.]
        img, bbox, scale = resize(im, bbox, size=self.img_size)
        img = np.transpose(img, (2, 0, 1))
        img = (img / 255.).astype(np.float32)

        img_info = {'img_width': self.img_size, 'img_height': self.img_size, 'scale': scale,
                    'img_name': img_name, 'ori_shape': im.size}

        return img, bbox, label, img_info



def resize(img, bbox=None, size=512):
    ori_W, ori_H = img.size

    img = img.resize((size, size))
    img = np.array(img)
    H, W, C = img.shape

    scale = [W / ori_W, H / ori_H]
    if bbox is not None:
        bbox = resize_bbox(bbox, scale)

    return img, bbox, scale

def resize_bbox(bbox, scale):
    gt_bbox = bbox.copy()
    gt_bbox[:, 0] = scale[0] * bbox[:, 0]
    gt_bbox[:, 1] = scale[1] * bbox[:, 1]
    gt_bbox[:, 2] = scale[0] * bbox[:, 2]
    gt_bbox[:, 3] = scale[1] * bbox[:, 3]
    return gt_bbox
