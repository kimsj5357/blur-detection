import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
from utils.Visualize import Visualize, vis_bbox
from model.trainer import Trainer
from dataset import resize

def get_anno(img_path):
    label_names = ['person', 'bicycle', 'car', 'bus', 'motorbike']
    anno_path = img_path.replace('JPEGImages', 'Annotations').replace('png', 'xml')
    anno = ET.parse(anno_path)

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
        label.append(label_names.index(name))
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    return bbox, label

def main():
    img_name = 'BD_Baidu_424.png'
    img_dir = './RESIDE/RTTS/JPEGImages'
    load_path = './checkpoints/dehaze/epoch_13.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = Visualize('dehaze')

    trainer = Trainer(vis).to(device)
    trainer.load(load_path)

    # img_arr = io.imread(os.path.join(img_dir, img_name))
    im = Image.open(os.path.join(img_dir, img_name))
    gt_bbox, gt_label = get_anno(os.path.join(img_dir, img_name))

    img_arr, gt_bbox, scale = resize(im, gt_bbox, 512)
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img = torch.tensor(img_arr[None, ...] / 255.).cuda().float()

    img_arr = img_arr.astype(np.uint8)
    dark_ch, A, raw_t, refined_t = trainer.dehaze(img_arr)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(np.transpose(img_arr, (1, 2, 0))); ax[0][0].set_axis_off(); ax[0][0].set_title('(a)', y=-0.11)
    ax[0][1].imshow(dark_ch, cmap='gray'); ax[0][1].set_axis_off(); ax[0][1].set_title('(b)', y=-0.11)
    ax[1][0].imshow(raw_t, cmap='gray'); ax[1][0].set_axis_off(); ax[1][0].set_title('(c)', y=-0.11)
    ax[1][1].imshow(np.transpose(refined_t, (1, 2, 0))); ax[1][1].set_axis_off(); ax[1][1].set_title('(d)', y=-0.11)
    plt.tight_layout()
    plt.show()

    trainer.eval()
    dcp_dehaze, pred_dehaze = trainer.get_dehaze(img)
    pred_dehaze = pred_dehaze.detach()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.transpose(img_arr, (1, 2, 0))); ax[0].set_axis_off(); ax[0].set_title('(a)', y=-0.12)
    ax[1].imshow(np.transpose(dcp_dehaze.cpu().numpy(), (1, 2, 0))); ax[1].set_axis_off(); ax[1].set_title('(b)', y=-0.12)
    ax[2].imshow(np.transpose(pred_dehaze[0].cpu().numpy(), (1, 2, 0))); ax[2].set_axis_off(); ax[2].set_title('(c)', y=-0.12)
    plt.tight_layout()
    plt.show()

    dehaze_res = trainer.faster_rcnn_dcp(pred_dehaze)
    ori_res = trainer.faster_rcnn_ori(img)
    results = torch.cat((dehaze_res, ori_res))

    trainer.faster_rcnn_dcp.use_preset('visualize')
    trainer.faster_rcnn_ori.use_preset('visualize')

    dehaze_bbox, dehaze_label, dehaze_score = trainer.faster_rcnn_dcp.get_vis_bboxes(pred_dehaze, dehaze_res)
    ori_bbox, ori_label, ori_score = trainer.faster_rcnn_ori.get_vis_bboxes(img, ori_res)
    fusion_bbox, fusion_label, fusion_score = trainer.faster_rcnn_dcp.get_vis_bboxes(img, results)

    fig, ax = plt.subplots(2, 2)
    ax[0][0] = vis_bbox(img_arr, gt_bbox, gt_label, ax=ax[0][0]); ax[0][0].set_axis_off(); ax[0][0].set_title('(a)', y=-0.12)
    ax[0][1] = vis_bbox(img_arr, fusion_bbox, fusion_label, fusion_score, ax=ax[0][1]); ax[0][1].set_axis_off(); ax[0][1].set_title('(b)', y=-0.12)
    ax[1][0] = vis_bbox(img_arr, ori_bbox, ori_label, ori_score, ax=ax[1][0]); ax[1][0].set_axis_off(); ax[1][0].set_title('(c)', y=-0.12)
    ax[1][1] = vis_bbox(pred_dehaze[0].cpu().numpy()*255, dehaze_bbox, dehaze_label, dehaze_score, ax=ax[1][1]); ax[1][1].set_axis_off(); ax[1][1].set_title('(d)', y=-0.12)
    plt.tight_layout()
    plt.show()


    print('Done')

if __name__ == '__main__':
    main()