import numpy as np
import torch as t

def iou_bboxes(bboxA, bboxB):
    low = np.s_[..., :2]
    high = np.s_[..., 2:]
    ious = iou(bboxA[:, None], bboxB[None], high, low)

    return ious

def iou(bboxA, bboxB, high, low):
    A, B = bboxA.copy(), bboxB.copy()
    A[high] += 1
    B[high] += 1
    intrs = (np.maximum(0, np.minimum(A[high], B[high])
                        - np.maximum(A[low], B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)


def loc2bbox(anchor, loc):
    if len(anchor) == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    anchor_w = anchor[:, 2] - anchor[:, 0]
    anchor_h = anchor[:, 3] - anchor[:, 1]
    anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_w
    anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_h

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * anchor_w[:, np.newaxis] + anchor_ctr_x[:, np.newaxis]
    ctr_y = dy * anchor_h[:, np.newaxis] + anchor_ctr_y[:, np.newaxis]
    w = np.exp(dw) * anchor_w[:, np.newaxis]
    h = np.exp(dh) * anchor_h[:, np.newaxis]

    res_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    res_bbox[:, 0::4] = ctr_x - 0.5 * w
    res_bbox[:, 1::4] = ctr_y - 0.5 * h
    res_bbox[:, 2::4] = ctr_x + 0.5 * w
    res_bbox[:, 3::4] = ctr_y + 0.5 * h

    return res_bbox

def bbox2loc(src_bbox, dst_bbox):
            # anchor    bbox
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (np.array): An image coordinate array whose shape is (R, 4). 'R' is the number of bounding boxes.
            These coordinates are 'p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}'.
        dst_bbox (np.array):  An image coordinate array whose shape is (R, 4).
            These coordinates are 'g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}'.

    Returns:
        np.array:
        Bounding box offsets and scales from :obj:`src_bbox`  to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    """
       computes the distance from ground-truth boxes to the given boxes, normed by their size
       :param ex_rois: n * 4 numpy array, given boxes
       :param gt_rois: n * 4 numpy array, ground-truth boxes
       :return: deltas: n * 4 numpy array, ground-truth boxes
       """

    # given a bounding box
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    # the target bounding box
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def _nms(dets, scores, thresh):
    keep = []
    if scores.shape[0] == 0 or scores is None:
        return keep

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print(scores, scores.shape)
    order = scores.argsort()[::-1]
    # order = t.argsort(scores, descending=True)

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep