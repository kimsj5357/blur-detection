import torch as t
import torch.nn as nn

from torchvision.ops import RoIPool
from torch.nn import functional as F

from model.frcnn.utils.array_tool import totensor

class RoIHead(nn.Module):
    def __init__(self,
                 classifier, n_class, roi_size, spatial_scale):
        super(RoIHead, self).__init__()

        # self.classifier = list(model.classifier)
        # del self.classifier[6]
        # if not opt.use_drop:
        #     del self.classifier[5]
        #     del self.classifier[2]
        # self.classifier = nn.Sequential(*self.classifier)
        self.classifier = classifier

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self._init_layers()
        self.init_weights()
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)
        self.roi_pool = nn.AdaptiveMaxPool2d(self.roi_size)

        self.roi_cls_loss = 0
        self.roi_bbox_loss = 0


    def _init_layers(self):
        self.cls_loc = nn.Linear(4096, self.n_class * 4)
        self.score = nn.Linear(4096, self.n_class)

    def init_weights(self):
        normal_init(self.cls_loc, std=0.001)
        normal_init(self.score, std=0.01)

    def forward(self, x, rois, roi_indices):
        # in case roi_indices is  ndarray
        # roi_indices = totensor(roi_indices).float()
        # rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1).contiguous()
        # NOTE: important: yx->xy
        # xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)

        x = self.classifier(pool)
        roi_loc = self.cls_loc(x)
        roi_cls = self.score(x)

        # print(x.shape, roi_cls_locs.shape, roi_scores.shape)

        loc = roi_loc.view(-1, self.n_class, 4)
        cls = roi_cls.view(-1, self.n_class, 1)
        roi_res = t.cat((loc, cls), dim=-1).contiguous()

        return roi_res

    def build_loss(self, roi_loc, roi_cls, gt_loc, gt_cls, roi_lamda=10):
        # if self.DEBUG:
        #     print('\nRoI LOSS')
        #     print('roi_loc shape:', roi_loc.shape)  # (128, 44)
        #     print('roi_cls shape:', roi_cls.shape)  # (128, 11)
        #     print('gt_loc shape:', gt_loc.shape)    # (128, 4)
        #     print('gt_cls shape:', gt_cls.shape)    # (128,)

        gt_cls = totensor(gt_cls).long().cuda()
        gt_loc = totensor(gt_loc).cuda()

        fg_cnt = t.sum(gt_cls.data.ne(0))
        bg_cnt = gt_cls.data.numel() - fg_cnt
        ce_weights = t.ones(roi_cls.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cls_loss = F.cross_entropy(roi_cls, gt_cls, weight=ce_weights)
        # cls_loss = F.cross_entropy(roi_cls, gt_cls)

        num_roi = roi_loc.size(0)
        roi_loc = roi_loc.view(-1, self.n_class, 4)
        roi_loc = roi_loc[t.arange(num_roi), gt_cls]

        mask = gt_cls > 0
        mask_loc_pred = roi_loc[mask]
        mask_loc_target = gt_loc[mask]

        N_reg = mask.float().sum()

        loc_loss = F.smooth_l1_loss(mask_loc_pred, mask_loc_target, size_average=False) / (N_reg + 1e-4)

        return cls_loss, loc_loss

    def loss(self, rpn_lamda=1):
        roi_loss = self.roi_cls_loss + self.roi_bbox_loss * rpn_lamda
        return roi_loss

def normal_init(module, mean=0, std=1., bias=0):
    nn.init.normal_(module.weight, mean, std)
    nn.init.constant_(module.bias, bias)
