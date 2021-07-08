import torch.nn.functional as F
import torch.nn as nn
from model.util import Matcher, BalancedPositiveNegativeSampler, roi_align
from model.box_ops import Boxcoder, box_iou, process_box, nms

class ROIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, num_trhesh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.mask_roi_pool = None
        self.mask_predictor = None

        self.proposal = Matcher