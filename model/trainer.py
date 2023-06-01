import os

import numpy as np
import torch
from torch import nn

from config import opt
from dataset import inverse_normalize
from model.frcnn.faster_rcnn import FasterRCNN
from model.ssd.ssd import build_ssd
from model.atmosphere_net import UNet
from model.discriminator import Discriminator
from model.frcnn.utils.array_tool import *
from dcp import DeHaze
from model.ssd.layers.modules import MultiBoxLoss

class Trainer(nn.Module):

    def __init__(self, opt, vis=None, mode='train', device='cuda',
                 DEBUG=False, DEBUG_VIS=False):
        super(Trainer, self).__init__()

        self.model = opt.model
        self.mode = mode

        self.epoch = 0
        self.dehaze = DeHaze(visualize=DEBUG)
        self.unet = UNet(in_dims=4, out_dims=3)

        if self.model == 'frcnn':
            self.det_ori = FasterRCNN(vis, DEBUG, DEBUG_VIS)
            self.det_dcp = FasterRCNN(vis, DEBUG, DEBUG_VIS)

            if self.mode == 'train':
                params = self.unet.get_optimizer_params(opt.lr, opt.weight_decay) \
                         + self.det_ori.get_optimizer_params() \
                         + self.det_dcp.get_optimizer_params()
                # params = self.det_ori.get_optimizer_params() + self.det_dcp.get_optimizer_params()
                self.optimizer = torch.optim.SGD(params, momentum=0.9)

        elif self.model == 'ssd512':
            self.det_ori = build_ssd(self.mode, size=512, num_classes=opt.num_class)
            self.det_dcp = build_ssd(self.mode, size=512, num_classes=opt.num_class)

            if self.mode == 'train':
                params = list(self.unet.parameters()) \
                         + list(self.det_ori.parameters()) \
                         + list(self.det_dcp.parameters())
                self.optimizer = torch.optim.AdamW(params, lr=opt.lr)

            self.criterion = MultiBoxLoss(opt.num_class, 0.5, True, 0, True, 3, 0.5, False, use_gpu=True)

        # if mode == 'train' and self.epoch == 0:
        #     self.initialize()

        self.losses = {}

        self.vis = vis
        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS


    def predict(self, imgs, img_info=None):
        dcp_dehaze, pred_dehaze = self.get_dehaze(imgs)
        results_1 = self.det_dcp(pred_dehaze, img_info=img_info)
        results_2 = self.det_ori(imgs, img_info=img_info)
        results = torch.cat((results_1, results_2))
        self.det_dcp.use_preset('visualize')
        bbox, label, score = self.det_dcp.get_vis_bboxes(imgs, results)
        del results_1, results_2, results
        return dcp_dehaze, pred_dehaze, [bbox], [label], [score]

    def predict_ssd(self, imgs, img_info=None):
        dcp_dehaze, pred_dehaze = self.get_dehaze(imgs)
        loc_dcp, conf_dcp, priors_dcp = self.det_dcp(pred_dehaze, img_info=img_info)
        out_dcp = self.det_dcp.detect(loc_dcp, self.det_dcp.softmax(conf_dcp), priors_dcp)
        loc_ori, conf_ori, priors_ori = self.det_ori(imgs, img_info=img_info)
        out_ori = self.det_ori.detect(loc_ori, self.det_ori.softmax(conf_ori), priors_ori)



    def forward(self, imgs, bboxes=None, labels=None, img_info=None):
        """Foward Faster R-CNN and calculate losses.

        :param imgs: (batch_size, C, H, W)
        :param img_info (dict)
        :param annos: (batch_size, num_gts, 5) in [x1, y1, x2, y2, label]
        :return: loss (dict)
        """

        # if len(imgs) != 1:
        #     print('Currently only batch size 1 is supported.')
        #     imgs = imgs[:1]


        self.optimizer.zero_grad()

        dcp_dehaze, pred_dehaze = self.get_dehaze(imgs)
        losses = self.losses

        if self.model == 'frcnn':
            self.det_dcp(pred_dehaze, bboxes, labels, img_info)
            dcp_loss = self.det_dcp.loss()

            self.det_ori(imgs, bboxes, labels, img_info)
            ori_loss = self.det_ori.loss()

            for key in dcp_loss.keys():
                losses[key] = (dcp_loss[key] + ori_loss[key]) / 2
            losses['det_loss'] = (dcp_loss['det_loss'] + ori_loss['det_loss']) / 2

        elif self.model == 'ssd512':
            out_dcp = self.det_dcp(pred_dehaze.detach())
            loss_l_dcp, loss_c_dcp = self.criterion(out_dcp, bboxes)

            out_ori = self.det_ori(imgs)
            loss_l_ori, loss_c_ori = self.criterion(out_ori, bboxes)

            losses['det_loss'] = loss_l_dcp + loss_c_dcp + loss_l_ori + loss_c_ori
            losses['det_loss'] = losses['det_loss'] / 2

        losses['total_loss'] = 10 * losses['dehaze_loss'] + losses['det_loss']
        # losses['total_loss'] = losses['frcnn_loss']
        losses['total_loss'].backward()

        self.optimizer.step()

        self.losses.update(losses)
        # print(self.losses)

        return losses

    def get_dehaze(self, imgs):
        dark_ch, A, raw_t, refined_t = [], [], [], []
        for i in range(imgs.size(0)):
            img_array = inverse_normalize(imgs[i].cpu().numpy()).astype(np.uint8)
            dark_ch_i, A_i, raw_t_i, refined_t_i = self.dehaze(img_array)
            dark_ch.append(dark_ch_i); A.append(A_i); raw_t.append(raw_t_i); refined_t.append(refined_t_i)
        dark_ch, A, raw_t, refined_t = np.array(dark_ch), np.array(A), np.array(raw_t), np.array(refined_t)
        dark_ch = np.expand_dims(dark_ch, axis=1)

        dark_ch = torch.from_numpy(dark_ch).to(imgs.device)    # (b, 1, h, w)
        img_dark = torch.cat((imgs, dark_ch), dim=1).float()   # (b, 4, h, w)

        pred_A = self.unet(img_dark)

        A = torch.from_numpy(A).float().to(imgs.device)
        refined_t = torch.from_numpy(refined_t.copy()).float().to(imgs.device)
        dcp_dehaze = self.dehaze.get_dehaze_tensor(inverse_normalize(imgs), A, refined_t).float()
        pred_dehaze = self.dehaze.get_dehaze_tensor(inverse_normalize(imgs), pred_A[0], refined_t).float()
        # pred_dehaze = self.dehaze._get_dehaze(inverse_normalize(img[0]), pred_A[0]).float()


        if self.training:
            dehaze_loss = self.unet.loss(pred_dehaze, dcp_dehaze)
            self.losses.update({'dehaze_loss': dehaze_loss})

        # pred_dehaze = pred_dehaze[None, ...].detach()


        # TODO: Discriminator
        # if self.training:
        #     valid = torch.autograd.Variable(torch.Tensor(1, 1).fill_(1.0), requires_grad=False).cuda()
        #     fake = torch.autograd.Variable(torch.Tensor(1, 1).fill_(0.0), requires_grad=False).cuda()
        #     real_loss = self.D.loss(self.D(dcp_dehaze), valid)
        #     fake_loss = self.D.loss(self.D(pred_dehaze), fake)
        #     d_loss = (real_loss + fake_loss) / 2
        #     self.losses.update({'dehaze_loss': d_loss})


        # pred_dehaze = dcp_dehaze[None, ...]
        return dcp_dehaze, pred_dehaze

    def get_loss(self):
        losses = {}
        for key, loss in self.losses.items():
            losses.update({key: loss.data.item()})
        return losses

    def vis_loss(self, i):
        for key, loss in self.losses.items():
            self.vis.plot(key, loss.view(-1))

    def initialize(self):
        print('Initializing weights...')

        self.unet.apply(weights_init)
        if self.model == 'frcnn':
            self.det_ori.apply(weights_init)
            self.det_dcp.apply(weights_init)
        elif self.model == 'ssd512':
            self.det_ori.extras.apply(weights_init)
            self.det_ori.loc.apply(weights_init)
            self.det_ori.conf.apply(weights_init)
            self.det_dcp.extras.apply(weights_init)
            self.det_dcp.loc.apply(weights_init)
            self.det_dcp.conf.apply(weights_init)


    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """

        params = []
        for key, value in dict(self.det.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': opt.lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': opt.lr, 'weight_decay': opt.weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        self.epoch += 1

        save_dict = dict()

        save_dict['model'] = self.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        # save_dict['vis_info'] = self.vis.state_dict()
        save_dict['epoch'] = self.epoch

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = os.path.join('checkpoints', opt.dataset, 'epoch_' + str(self.epoch) + '.pth')

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        self.epoch = state_dict['epoch']

        # loss = state_dict['other_info']['loss']
        # return loss


def xavier(param):
    nn.init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()