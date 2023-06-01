import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchsummary import summary

from dataset import DehazeDataset
from utils.Visualize import Visualize
from model.trainer import Trainer
from utils.eval_tool import evaluate

import gc
gc.collect()
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Dark Channel Prior')
    parser.add_argument('--root_dir', default='./RESIDE/RTTS')
    parser.add_argument('--model', default='frcnn', choices=['frcnn', 'ssd512'])
    parser.add_argument('--num_class', default=5)
    parser.add_argument('--epoch', default=300)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=1e-8)
    parser.add_argument('--gamma', default=0.1)

    parser.add_argument('--checkpoint', default='./checkpoints')
    parser.add_argument('--resume', default=False)
    parser.add_argument('--load-path', default='./checkpoints/dehaze/epoch_34.pth')
    parser.add_argument('--use-visdom', default=True)
    parser.add_argument('--env', default='dehaze')
    return parser.parse_args()

def fix_seed(device):
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    if device == 'cuda':
        torch.cuda.manual_seed_all(4321)

def main():
    args = parse_args()

    DEBUG = False
    DEBUG_VIS = False
    if DEBUG_VIS:
        args.env = args.env + '-DEBUG'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = Visualize(args.env) if args.use_visdom else None

    fix_seed(device)

    # TODO: option change
    save_dir = os.path.join(args.checkpoint, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    dataset = DehazeDataset(args.root_dir)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=8, pin_memory=True)


    trainer = Trainer(args, vis, DEBUG=DEBUG, DEBUG_VIS=DEBUG_VIS)
    trainer = nn.DataParallel(trainer)
    trainer = trainer.to(device)

    if args.resume:
        trainer.load(args.load_path)


    # start_epoch = trainer.epoch
    start_epoch = trainer.module.epoch
    for epoch in range(start_epoch, args.epoch):
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, data in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch} ")

            if args.model == 'frcnn':
                img, bbox, label, img_info = data
                img = img.to(device)
                bbox = bbox.to(device)
                label = label.to(device)
                input = (img, bbox, label, img_info)

            elif args.model == 'ssd512':
                img, gt_annos, img_info = data
                img = img.to(device)
                gt_annos = [anno.to(device) for anno in gt_annos]
                input = (img, gt_annos, None, img_info)

            trainer(*input)

            if args.use_visdom:
                # trainer.vis_loss(i)
                trainer.module.vis_loss(i)

            if i % 100 == 0:
                print(trainer.module.get_loss())
                if args.use_visdom:
                    trainer.eval()

                    with torch.no_grad():
                        dcp_dehaze, pred_dehaze, _bboxes, _labels, _scores = trainer.predict(img, img_info)

                        vis.imshow('ori_img', img[0])
                        vis.imshow('dcp_dehaze', dcp_dehaze)
                        vis.imshow('pred_dehaze', pred_dehaze)

                        gt_img = vis.visdom_bbox(img[0].cpu().numpy()*255,
                                                 gt_bbox[0].cpu().numpy(),
                                                 gt_label[0].cpu().numpy())
                        vis.imshow('gt_img', gt_img)
                        pred_img = vis.visdom_bbox(pred_dehaze[0].cpu().numpy()*255,
                                                   _bboxes[0],
                                                   _labels[0],
                                                   scores=_scores[0])
                        vis.imshow('pred_img', pred_img)
                        del dcp_dehaze, pred_dehaze, _bboxes, _labels, _scores


                    trainer.train()



        save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
        trainer.save(save_path=save_path)

        trainer.eval()
        eval_result = evaluate(val_loader, trainer, visualize=True, vis=Visualize(args.env))
        trainer.train()
        print('MAE:{:.5f}'.format(eval_result['MAE']))
        print('mAP:{:.5f}'.format(eval_result['map']))


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    annos = []
    info = []
    for sample in batch:
        imgs.append(torch.FloatTensor(sample[0]))
        annos.append(torch.FloatTensor(sample[1]))
        info.append(sample[2])
    return torch.stack(imgs, 0), annos, info


if __name__ == '__main__':
    main()
