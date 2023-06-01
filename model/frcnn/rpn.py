from torch import nn
import torch.nn.functional as F

from model.frcnn.utils.array_tool import *
from model.frcnn.utils.creator_tools import ProposalCreator, AnchorTargetCreator


class RPN(nn.Module):
    def __init__(self, vis, device,
                 DEBUG=False, DEBUG_VIS=False,
                 in_channels=512, mid_channels=512, ratios=[0.5, 1., 2.],
                 anchor_scales=[8, 16, 32], feat_stride=16):
        super(RPN, self).__init__()

        self.vis = vis
        self.device = device
        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS

        self.anchor_base = base_anchor_generator(base_size=feat_stride, ratios=ratios, anchor_scales=anchor_scales)
        self.anchor_target_creator = AnchorTargetCreator(vis=vis, DEBUG=DEBUG, DEBUG_VIS=DEBUG_VIS)
        self.proposal_layer = ProposalCreator(self)

        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.ratios = ratios

        self.num_anchors = len(ratios) * len(anchor_scales)
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self._init_layers()
        self.init_weights()

        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(self.in_channels, self.mid_channels, 3, 1, 1)
        self.rpn_cls = nn.Conv2d(self.mid_channels, self.num_anchors * 2, 1, 1, 0)
        self.rpn_loc = nn.Conv2d(self.mid_channels, self.num_anchors * 4, 1, 1, 0)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_loc, std=0.01)

    def forward(self, x, gt_bboxes=None, img_info=None):
        n, _, hh, ww = x.shape

        if self.DEBUG:
            print()
            print('RPN')
            print('features shape:', x.shape)

        # (9 * hh * ww, 4) [x, y, x, y]
        anchor = anchor_generator(self.anchor_base, self.feat_stride, hh, ww)

        if self.DEBUG:
            print('anchors:', anchor.shape[0])
            print('max anchor[:,2]:', np.max(anchor[:, 2]), '  max anchor[:, 3]:', np.max(anchor[:, 3]))

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for i, box in enumerate(anchor):
                if i % 400 == 0:
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                               color='red', fill=False, linewidth=1))
            plt.xlim(np.min(anchor[:, 0]), np.max(anchor[:, 2]))
            plt.ylim(np.max(anchor[:, 3]), np.min(anchor[:, 1]))
            plt.title(str(len(anchor)) + ' anchors')
            plt.show()

        if self.DEBUG_VIS and self.vis is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_xlim(np.min(anchor[:, 0]), np.max(anchor[:, 2]))
            ax.set_ylim(np.max(anchor[:, 3]), np.min(anchor[:, 1]))
            ax.set_title(str(len(anchor)) + ' anchors')
            vis_anchor = self.vis.visdom_bbox(None, anchor[0::400], ax=ax)
            self.vis.imshow('anchor', vis_anchor)

        x = self.rpn_conv(x)  # (1, 512, hh, ww)
        x = F.relu(x, inplace=True)  # (1, 512, hh, ww)
        pred_loc = self.rpn_loc(x)  # (1,  36, hh, ww)
        pred_cls = self.rpn_cls(x)  # (1,  18, hh, ww)

        if self.DEBUG:
            print('pred_loc shape:', pred_loc.shape)
            print('pred_cls shape', pred_cls.shape)

        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (1, 9 * hh * ww, 4)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous()  # (1, hh, ww, 18)
        pred_softmax_cls = F.softmax(pred_cls.view(n, hh, ww, self.num_anchors, 2), dim=4)  # (1, hh, ww, 9, 2)
        pred_fg_cls = pred_softmax_cls[:, :, :, :, 1].contiguous()  # (1, hh, ww, 9)
        pred_fg_cls = pred_fg_cls.view(n, -1)  # (1, 9 * hh * ww)
        pred_cls = pred_cls.view(n, -1, 2)  # (1, 9 * hh * ww, 2)

        if self.DEBUG:
            print('pred_loc shape:', pred_loc.shape)
            print('pred_cls shape:', pred_cls.shape)
            print('pred_fg_cls shape:', pred_fg_cls.shape)

        roi = self.proposal_layer(  # (2000, 4) 최대 2000
            pred_loc[0].cpu().data.numpy(),
            pred_fg_cls[0].cpu().data.numpy(),
            anchor,
            img_info)
        roi_indices = np.zeros((len(roi),), dtype=np.int32)
        # pred_obj = pred_cls[:, :, 1]
        # rois = list()
        # roi_indices = list()
        # for i in range(n):
        #     roi = self.proposal_layer(  # (2000, 4) 최대 2000
        #         pred_loc[i].cpu().data.numpy(),
        #         pred_fg_cls[i].cpu().data.numpy(),
        #         anchor,
        #         img_info)
        #     batch_index = i * np.ones((len(roi),), dtype=np.int32)  # (2000,)
        #     rois.append(roi)
        #     roi_indices.append(batch_index)
        #
        # rois = np.concatenate(rois, axis=0)  # (2000, 4) [x, y, x, y]
        # roi_indices = np.concatenate(roi_indices, axis=0)  # (2000,)   [0, 0, ..., 0]

        if self.DEBUG:
            print('rois shape', roi.shape)
            print('roi_indices shape', roi_indices.shape)

        if self.training:
            gt_bbox = tonumpy(gt_bboxes)[0]
            target_loc, target_cls = self.anchor_target_creator(gt_bbox, anchor, img_info)

            if self.DEBUG:
                print()
                print('target_loc shape:', target_loc.shape, np.max(target_loc))
                print('target_cls shape:', target_cls.shape)

                from model.frcnn.utils.bbox_tools import loc2bbox
                W, H = self.feat_stride * ww, self.feat_stride * hh
                pred_bbox = tonumpy(pred_loc)
                pred_bbox = loc2bbox(anchor, pred_bbox[0])
                mask = target_cls > 0
                pred_bbox = pred_bbox[mask]
                fig, ax = plt.subplots()
                for i, box in enumerate(pred_bbox):
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                               color='red', fill=False, linewidth=1))
                plt.xlim(0, W)
                plt.ylim(H, 0)
                plt.title(str(len(pred_bbox)) + ' proposals')
                plt.show()

            if self.DEBUG_VIS:
                from model.frcnn.utils.bbox_tools import loc2bbox
                W, H = self.feat_stride * ww, self.feat_stride * hh
                pred_bbox = tonumpy(pred_loc)
                pred_bbox = loc2bbox(anchor, pred_bbox[0])
                mask = target_cls > 0
                pred_bbox = pred_bbox[mask]

                fig, ax = plt.subplots()
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)
                ax.set_title(str(len(pred_bbox)) + ' predict proposals')
                vis_proposals = self.vis.visdom_bbox(None, pred_bbox, ax=ax)
                self.vis.imshow('proposals', vis_proposals)

            self.rpn_cls_loss, self.rpn_bbox_loss = self.build_loss(pred_loc[0], pred_cls[0], target_loc, target_cls)

        roi = totensor(roi).float()
        roi_indices = totensor(roi_indices).float().view(-1, 1)
        rois = t.cat((roi, roi_indices), dim=-1)

        return rois

    def build_loss(self, pred_loc, pred_cls, target_loc, target_cls):
        if self.DEBUG:
            print('\nRPN LOSS')
            print('pred_loc shape:', pred_loc.shape)
            print('pred_cls shape:', pred_cls.shape)
            print('target_loc shape:', target_loc.shape)
            print('target_cls shape:', target_cls.shape)

        # cls loss
        gt_rpn_cls = totensor(target_cls).long().cuda()
        pred_rpn_cls = pred_cls

        rpn_cls_loss = F.cross_entropy(pred_rpn_cls, gt_rpn_cls, ignore_index=-1)


        gt_rpn_loc = totensor(target_loc)
        pred_rpn_loc = pred_loc

        mask = gt_rpn_cls > 0
        # print(mask.shape)
        mask_gt_loc = gt_rpn_loc[mask]
        mask_pred_loc = pred_rpn_loc[mask]

        N_reg = mask.float().sum()


        rpn_loc_loss = F.smooth_l1_loss(mask_pred_loc, mask_gt_loc.cuda(), size_average=False) / (N_reg + 1e-4)


        return rpn_cls_loss, rpn_loc_loss

    def loss(self, rpn_lamda=1):
        rpn_loss = self.rpn_cls_loss + self.rpn_bbox_loss * rpn_lamda
        return rpn_loss


def base_anchor_generator(base_size=16,
                          ratios=[0.5, 1., 2.],
                          anchor_scales=[8, 16, 32]):
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    for x in range(len(ratios)):
        for y in range(len(anchor_scales)):
            h = base_size * anchor_scales[y] * np.sqrt(ratios[x])
            w = base_size * anchor_scales[y] * np.sqrt(1. / ratios[x])

            index = x * len(anchor_scales) + y
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.


    return anchor_base


def anchor_generator(anchor_base, feat_stride, height, width):
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    # print(shift_x.shape, shift_y.shape)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # print(shift_x.shape, shift_y.shape)
    shift = np.vstack((shift_x.ravel(), shift_y.ravel(),
                       shift_x.ravel(), shift_y.ravel())).transpose()
    # print(shift.shape)

    A = len(anchor_base)
    K = len(shift)

    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # print('anchor', anchor.shape)

    return anchor



def normal_init(module, mean=0, std=1., bias=0):
    nn.init.normal_(module.weight, mean, std)
    nn.init.constant_(module.bias, bias)
