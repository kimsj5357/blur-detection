import numpy as np

import torch
from torchvision.ops import nms
from model.frcnn.utils.bbox_tools import bbox2loc, iou_bboxes, loc2bbox

from config import opt

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self, vis=None, DEBUG=False, DEBUG_VIS=False,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

        self.vis = vis
        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=[0., 0., 0., 0.],
                 loc_normalize_std=[0.1, 0.1, 0.2, 0.2]):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (np.array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (np.array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (np.array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bounding boxes.
            loc_normalize_std (tuple of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (np.array, np.array, np.array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_roi = len(roi)

        # self.DEBUG = True
        if self.DEBUG:
            print('roi shape:', roi.shape)
            print('bbox shape:', bbox.shape)
            print('label shape:', label.shape)

        roi = np.concatenate((roi, bbox), axis=0)

        if self.DEBUG:
            print('roi shape:', roi.shape)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = iou_bboxes(roi, bbox)

        argmax_iou = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        if self.DEBUG:
            print('max iou', max_iou[:n_roi])
            print('max max_iou:', np.max(max_iou[:n_roi]))

        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[argmax_iou] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou > self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        keep_bbox = bbox[argmax_iou[keep_index]]

        if self.DEBUG:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for i, box in enumerate(bbox):
                x1, y1, x2, y2 = box
                # print(box)
                ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                           color='blue', fill=False, linewidth=2))

            for i, box in enumerate(keep_bbox[:pos_roi_per_this_image]):
                x1, y1, x2, y2 = box
                ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                           color='red', fill=False, linewidth=1))
            plt.xlim(0, np.max(roi[:, 2]))
            plt.ylim(np.max(roi[:, 3]), 0)
            plt.title(str(pos_roi_per_this_image) + ' pos rois(red), ' + str(len(bbox)) + ' bbox(blue)')
            plt.show()

        if self.DEBUG_VIS and self.vis is not None:
            pos = keep_bbox[:pos_roi_per_this_image]
            bbox_and_pos = np.concatenate((bbox, pos), axis=0)
            tmp_label = np.zeros(len(bbox_and_pos), dtype=np.int)
            tmp_label[:len(bbox)].fill(1)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_xlim(0, np.max(roi[:, 2]))
            ax.set_ylim(np.max(roi[:, 3]), 0)
            ax.set_title(str(pos_roi_per_this_image) + ' pos rois(blue), ' + str(len(bbox)) + ' bbox(orange)')
            vis_pos_rois = self.vis.visdom_bbox(None, bbox_and_pos, labels=tmp_label, ax=ax)
            self.vis.imshow('pos rois', vis_pos_rois)

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, keep_bbox)
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))
        gt_roi_loc = np.nan_to_num(gt_roi_loc)

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self, vis=None,
                 DEBUG=False, DEBUG_VIS=False,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

        self.vis = vis
        self.DEBUG = DEBUG
        self.DEBUG_VIS = DEBUG_VIS

    def __call__(self, bbox, anchor, img_info=None):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of gt bounding boxes.

        Args:
            bbox (np.array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (np.array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_info (dict):

        Returns:
            (np.array, np.array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        # self.DEBUG = True
        if img_info is not None:
            img_W = img_info['img_width'].item()
            img_H = img_info['img_height'].item()
        else:
            img_W, img_H = opt.img_size, opt.img_size


        n_anchor = len(anchor)
        inside_index = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= img_W) &
            (anchor[:, 3] <= img_H))[0]

        if self.DEBUG:
            print('max anchor[:,2]:', np.max(anchor[:, 2]), '  max anchor[:, 3]:', np.max(anchor[:, 3]))
            print('inside_index shape:', inside_index.shape)

        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # self.DEBUG = True
        if self.DEBUG:
            print('inside anchor shape:', anchor.shape)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            pos_anchor = anchor[np.where(label == 1)[0]]
            for i, box in enumerate(pos_anchor):
                x1, y1, x2, y2 = box
                ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                           color='red', fill=False, linewidth=1))
            for i, box in enumerate(bbox):
                x1, y1, x2, y2 = box
                # print(box)
                ax.add_patch(plt.Rectangle((int(x1), int(y1)), (int(x2) - int(x1)), (int(y2) - int(y1)),
                                           color='blue', fill=False, linewidth=2))
            plt.xlim(0, img_W)
            plt.ylim(img_H, 0)
            plt.title(str(len(pos_anchor)) + ' pos anchors, ' + str(len(bbox)) + ' bbox')
            plt.show()
        # self.DEBUG = False
        if self.DEBUG_VIS and self.vis is not None:
            pos_anchor = anchor[np.where(label == 1)[0]]
            pos_and_bbox = np.concatenate((pos_anchor, bbox), axis=0)
            tmp_label = np.zeros(len(pos_and_bbox), dtype=np.int)
            tmp_label[len(pos_anchor):].fill(1)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_xlim(0, img_W)
            ax.set_ylim(img_H, 0)
            ax.set_title(str(len(pos_anchor)) + ' pos anchors, ' + str(len(bbox)) + ' bbox')
            vis_pos_anchor = self.vis.visdom_bbox(None, pos_and_bbox, labels=tmp_label, ax=ax)
            self.vis.imshow('pos anchor', vis_pos_anchor)


        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])


        if self.DEBUG:
            print('max loc[:,2]:', np.max(loc[:, 2]), '  max loc[:, 3]:', np.max(loc[:, 3]))
            print('loc shape:', loc.shape)

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        # u, indices = np.unique(label, return_counts=True)
        # print(u, indices)

        if self.DEBUG:
            print('label shape', label.shape)
            print('loc shape', loc.shape)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1


        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        if self.DEBUG:
            print('rpn: max max_ious', np.max(max_ious))
            print('rpn: num_positive', np.sum(label == 1))
            print('rpn: num_negative', np.sum(label == 0))

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = iou_bboxes(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16,
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

        self.DEBUG = self.parent_model.DEBUG

    def __call__(self, loc, score, anchor, img_info=None):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (np.array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (np.array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (np.array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_info (dict):

        Returns:
            np.array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms


        if img_info is not None:
            img_W = img_info['img_width'].item()
            img_H = img_info['img_height'].item()
            scale = [img_info['scale'][0].item(), img_info['scale'][1].item()]
        else:
            img_W, img_H = opt.img_size, opt.img_size
            scale = [1., 1.]

        # Convert anchors into proposal via bbox transformations.

        if self.DEBUG:
            print()
            print('Proposal layer')
            print('loc shape', loc.shape)
            print('score shape', score.shape)
            print('anchor shape', anchor.shape)

            print('max anchor[:,2]:', np.max(anchor[:, 2]), '  max anchor[:, 3]:', np.max(anchor[:, 3]))

        roi = loc2bbox(anchor, loc)

        if self.DEBUG:
            print('roi shape', roi.shape)
            print('max roi[:, 2]:', np.max(roi[:, 2]), '  max roi[:, 3]:', np.max(roi[:, 3]))

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_W)
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_H)
        # print(roi.shape)

        if self.DEBUG:
            print('After slicing')
            print('max roi[:, 2]:', np.max(roi[:, 2]), '  max roi[:, 3]:', np.max(roi[:, 3]))


        # Remove predicted boxes with either height or width < threshold.
        # min_size = self.min_size * scale
        min_size = [self.min_size * scale[0], self.min_size * scale[1]]
        ws = roi[:, 2] - roi[:, 0]
        hs = roi[:, 3] - roi[:, 1]

        if self.DEBUG:
            print('min size:', min_size)
            print('ws shape:', ws.shape, 'max:', np.max(ws))
            print('hs shape:', hs.shape, 'max:', np.max(hs))


        keep = np.where((hs >= min_size[1]) & (ws >= min_size[0]))[0]
        roi = roi[keep, :]
        score = score[keep]

        if self.DEBUG:
            print('keep shape', keep.shape)
            print('roi shape', roi.shape)
            print('score shape', score.shape)


        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        if self.DEBUG:
            print('Before nms')
            print('roi shape', roi.shape)
            print('score shape', score.shape)

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        score = score[keep.cpu().numpy()]

        if self.DEBUG:
            print('After nms')
            print('roi shape', roi.shape)
            print('score shape', score.shape)

        return roi
