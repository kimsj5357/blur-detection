import numpy as np

import torch as t
from torch import nn
from torch.nn import functional as F

from config import opt
from model.frcnn.vgg import vgg_16
from model.frcnn.rpn import RPN
from model.frcnn.roi_head import RoIHead
from model.frcnn.utils.creator_tools import ProposalTargetCreator
from model.frcnn.utils.array_tool import tonumpy, totensor
from model.frcnn.utils.bbox_tools import loc2bbox, _nms

def nograd(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    def __init__(self, vis=None,
                 DEBUG=False, DEBUG_VIS=False,
                 loc_normalize_mean=[0., 0., 0., 0.],
                 loc_normalize_std=[0.1, 0.1, 0.2, 0.2]):
        super(FasterRCNN, self).__init__()

        device = t.device("cuda")
        self.vis = vis

        if opt.net == 'vgg16':
            extractor, classifier = vgg_16(device)

        self.num_class = opt.num_classes


        self.extractor = extractor
        self.rpn = RPN(vis, device, DEBUG=DEBUG, DEBUG_VIS=DEBUG_VIS)
        self.roi_head = RoIHead(
            classifier,
            self.num_class + 1, # class + background
            roi_size=7,
            spatial_scale=(1. / self.rpn.feat_stride))

        self.proposal_target_creator = ProposalTargetCreator(vis, DEBUG, DEBUG_VIS)

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.use_adam = opt.use_adam

        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS


    def set_DEBUG(self, DEBUG, DEBUG_VIS):
        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS
        self.rpn.DEBUG = DEBUG
        self.rpn.DEBUG_VIS = DEBUG_VIS
        self.proposal_target_creator.DEBUG = DEBUG
        self.proposal_target_creator.DEBUG_VIS = DEBUG_VIS
        self.rpn.anchor_target_creator.DEBUG = DEBUG
        self.rpn.anchor_target_creator.DEBUG_VIS = DEBUG_VIS

    @property
    def n_class(self):
        return self.num_class + 1

    def forward(self, imgs, bboxes=None, labels=None, img_info=None):
        _, C, H, W = imgs.shape

        if bboxes is None:
            self.eval()
        else:
            self.train()

        if self.DEBUG and self.training:
            print('image shape:', imgs.shape)
            print('bboxes shape', bboxes.shape)
            print('img_info', img_info)

            import matplotlib.pyplot as plt
            from utils.Visualize import vis_bbox
            from dataset import inverse_normalize
            fig, ax = plt.subplots()
            vis_bbox(inverse_normalize(tonumpy(imgs[0])), tonumpy(bboxes[0]), labels=tonumpy(labels)[0], ax=ax)
            plt.xlim(0, W)
            plt.ylim(H, 0)
            plt.title(str(len(tonumpy(bboxes[0]))) + ' bbox')
            plt.show()
        if self.DEBUG_VIS and self.vis is not None and self.training:
            from dataset import inverse_normalize
            gt_img = self.vis.visdom_bbox(inverse_normalize(tonumpy(imgs[0])),
                                          tonumpy(bboxes[0]),
                                          tonumpy(labels[0]))
            self.vis.imshow('gt_img', gt_img)


        features = self.extractor(imgs)
        # (1, 512, hh, ww)

        rois = self.rpn(features, bboxes, img_info)
        roi_indices = rois[:, 4].view(-1)
        rois = rois[:, :4]
        # rois          (2000, 4) [x, y, x, y]
        # roi_indices   (2000,)   [0, 0, ..., 0]

        if self.DEBUG and self.training:
            print('rois shape:', rois.shape)
            print('roi_indices shape:', roi_indices.shape)
            print()

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for i, box in enumerate(tonumpy(rois)):
                if i % 10 == 0:
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                               color='red', fill=False, linewidth=1))
            plt.xlim(0, W)
            plt.ylim(H, 0)
            plt.title('After RPN: ' + str(rois.shape[0]) + ' rois')
            plt.show()

        if self.DEBUG_VIS and self.vis is not None and self.training:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_title('After RPN: ' + str(rois.shape[0]) + ' rois')
            vis_rois = self.vis.visdom_bbox(None, tonumpy(rois)[0::10], ax=ax)
            self.vis.imshow('rois_rpn', vis_rois)


        if self.training:
            rois, gt_roi_loc, gt_roi_label = self.proposal_target_creator(tonumpy(rois), tonumpy(bboxes[0]), tonumpy(labels[0]))
            rois = totensor(rois).float()
            # rois          (128, 4) [x, y, x, y]
            # gt_roi_loc    (128, 4)
            # gt_roi_label  (128,)
            roi_indices = t.zeros(len(rois)).cuda().float()
            # roi_indices   (128,)   [0, 0, ..., 0]


            if self.DEBUG:
                fig, ax = plt.subplots()
                for i, box in enumerate(tonumpy(rois)):
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                               color='red', fill=False, linewidth=1))
                plt.xlim(0, W)
                plt.ylim(H, 0)
                plt.title('After proposal_target_layer: ' + str(rois.shape[0]) + ' rois')
                plt.show()
            if self.DEBUG_VIS and self.vis is not None:
                fig, ax = plt.subplots()
                ax.set_xlim(0, W)
                ax.set_ylim(H, 0)
                ax.set_title('After proposal_target_layer: ' + str(len(rois)) + ' rois')
                # vis_rois = self.vis.visdom_bbox(None, rois, ax=ax)
                rois_and_bbox = np.concatenate((tonumpy(rois), tonumpy(bboxes[0])), axis=0)
                tmp_label = np.zeros(len(rois_and_bbox)).astype(np.int32)
                tmp_label[rois.shape[0]:].fill(1)
                vis_rois = self.vis.visdom_bbox(None, rois_and_bbox, tmp_label, ax=ax)
                self.vis.imshow('rois_ptl', vis_rois)

        roi_res = self.roi_head(features, rois, roi_indices)
        roi_cls = roi_res[:, :, 4].reshape(-1, self.n_class)
        roi_loc = roi_res[:, :, :4].reshape(-1, self.n_class * 4)
        # roi_cls   (128, 11)
        # roi_loc   (128, 4 * 11)

        if self.DEBUG or self.DEBUG_VIS and self.training:
            pred = tonumpy(roi_loc)

            mean = np.tile(self.loc_normalize_mean, self.n_class)
            std = np.tile(self.loc_normalize_std, self.n_class)

            roi_cls_loc = (pred * std + mean)
            roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 4)
            roi = np.tile(tonumpy(rois), self.n_class).reshape(roi_cls_loc.shape)
            cls_bbox = loc2bbox(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = np.clip(cls_bbox[:, 0::2], 0, W)
            cls_bbox[:, 1::2] = np.clip(cls_bbox[:, 1::2], 0, H)

            prob = F.softmax(totensor(roi_cls), dim=1)
            prob = tonumpy(prob)

            self.nms_thresh = 0.3
            self.score_thresh = 0.1
            bbox = list(); label = list(); score = list()
            for l in range(1, self.n_class):
                cls_bbox_l = cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
                prob_l = prob[:, l]
                mask = prob_l > self.score_thresh
                cls_bbox_l = cls_bbox_l[mask]
                prob_l = prob_l[mask]
                keep = _nms(cls_bbox_l, prob_l, self.nms_thresh)
                bbox.append(cls_bbox_l[keep])
                label.append((l - 1) * np.ones((len(keep),)))
                score.append(prob_l[keep])
            bbox = np.concatenate(bbox, axis=0).astype(np.float32)
            label = np.concatenate(label, axis=0).astype(np.int32)
            score = np.concatenate(score, axis=0).astype(np.float32)

        if self.DEBUG and self.training:
            fig, ax = plt.subplots()
            from utils.Visualize import vis_bbox
            vis_bbox(inverse_normalize(tonumpy(imgs[0])), bbox, labels=label, scores=score, ax=ax)
            plt.xlim(0, W)
            plt.ylim(H, 0)
            plt.title('Predict bbox: ' + str(len(bbox)) + '(score threshold: ' + str(self.score_thresh) + ')')
            plt.show()

        if self.DEBUG_VIS and self.vis is not None and self.training:
            fig, ax = plt.subplots()
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_title('Predict bbox: ' + str(len(bbox)) + '(score threshold: ' + str(self.score_thresh) + ')')
            vis_pred = self.vis.visdom_bbox(inverse_normalize(tonumpy(imgs[0])), bbox, label, score, ax=ax)
            self.vis.imshow('pred_bbox', vis_pred)

        if self.training:
            self.roi_head.roi_cls_loss, self.roi_head.roi_bbox_loss = \
                self.roi_head.build_loss(roi_loc, roi_cls, gt_roi_loc, gt_roi_label)

        res = t.cat((roi_cls, roi_loc, rois, roi_indices.reshape(-1, 1)), dim=-1)
        # res: (128, n_class * 5 + 5) = (128, n_class) + (128, n_class * 4) + (128, 4) + (128, 1)

        return res


    def loss(self):

        losses = {}

        rpn_loss = self.rpn.loss()
        losses.update({'rpn_cls_loss': self.rpn.rpn_cls_loss, 'rpn_bbox_loss': self.rpn.rpn_bbox_loss})

        roi_loss = self.roi_head.loss()
        losses.update({'roi_cls_loss': self.roi_head.roi_cls_loss, 'roi_bbox_loss': self.roi_head.roi_bbox_loss})

        total_loss = rpn_loss + roi_loss
        losses.update({'det_loss': total_loss})


        return losses


    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            # if prob_l.shape[0] > 0:
            #     self.vis.vis.text('score_'+str(l)+': '+str(tonumpy(prob_l))+'\n'+str(prob_l.shape), 'score')
            keep = _nms(tonumpy(cls_bbox_l), tonumpy(prob_l), self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    @nograd
    def predict(self, imgs, img_infos=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
        else:
            self.use_preset('evaluate')
        bboxes = list()
        labels = list()
        scores = list()
        for img, img_info in zip(imgs, img_infos):
            img = totensor(img[None]).float()
            _, C, H, W = img.shape

            results = self(img, img_info=img_info)
            bbox, label, score = self.get_vis_bboxes(img, results)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        # self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_vis_bboxes(self, img, results):
        _, C, H, W = img.shape

        roi_scores = results[:, :self.n_class]
        roi_cls_loc = results[:, self.n_class:5 * self.n_class]
        rois = results[:, 5 * self.n_class:5 * self.n_class + 4]

        roi_score = roi_scores.data
        roi_cls_loc = roi_cls_loc.data
        roi = totensor(rois)

        # Convert predictions to bounding boxes in image coordinates.
        # Bounding boxes are scaled to the scale of the input images.
        mean = t.Tensor(self.loc_normalize_mean).cuda(). \
            repeat(self.n_class)[None]
        std = t.Tensor(self.loc_normalize_std).cuda(). \
            repeat(self.n_class)[None]

        roi_cls_loc = (roi_cls_loc * std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),
                            tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.n_class * 4)
        # clip bounding box
        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=W)
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=H)

        prob = F.softmax(totensor(roi_score), dim=1)

        bbox, label, score = self._suppress(cls_bbox, prob)
        return bbox, label, score

    def get_optimizer_params(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """

        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': self.lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': self.lr, 'weight_decay': self.weight_decay}]
        # if self.use_adam:
        #     self.optimizer = t.optim.Adam(params)
        # else:
        #     self.optimizer = t.optim.SGD(params, momentum=0.9)
        # return self.optimizer
        return params
