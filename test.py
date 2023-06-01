import random
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from config import opt
from utils.Visualize import Visualize
from dataset import DehazeDataset
from model.trainer import Trainer
from utils.eval_tool import evaluate

def fix_seed(device):
    random.seed(4321)
    np.random.seed(4321)
    torch.manual_seed(4321)
    if device == 'cuda':
        torch.cuda.manual_seed_all(4321)

def frcnn_load(trainer, load_path):
    state_dict = torch.load(load_path)
    if 'model' in state_dict:
        trainer.faster_rcnn_ori.load_state_dict(state_dict['model'])
    if 'optimizer' in state_dict:
        trainer.faster_rcnn_ori.optimizer.load_state_dict(state_dict['optimizer'])
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Dark Channel Prior')
    parser.add_argument('--root_dir', default='./RESIDE/RTTS')
    parser.add_argument('--model', default='frcnn', choices=['frcnn', 'ssd512'])
    parser.add_argument('--load-path', default='./checkpoints/dehaze/epoch_13.pth')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = Visualize('dehaze')
    fix_seed(device)

    dataset = DehazeDataset(args.root_dir)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=8, pin_memory=True)

    trainer = Trainer(args, vis, mode='test').to(device)
    trainer.load(args.load_path)

    eval_result = evaluate(val_loader, trainer, visualize=False, vis=vis, sleep=0)
    print('MAE:{:.4f}'.format(eval_result['MAE']))
    print('mAP:{:.4f}'.format(eval_result['map']))
    for i, label in enumerate(dataset.label_names):
        print('{}: {:.4f}'.format(label, eval_result['ap'][i]))


if __name__ == '__main__':
    main()