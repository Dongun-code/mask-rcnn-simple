import torch.nn as nn
import torch.nn.functional as F
import torch

from model.util import Matcher, BalancedPositiveNegativeSampler
from model.box_ops import Boxcoder, box_iou, nms, process_box


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride = 1, padding = 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)

        for i in self.children():
            nn.init.normal_(i.weight, std=0.01)
            nn.init.constant_(i.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)

        return logits, bbox_reg
            

class RPNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()

        self.anchor_generator = anchor_generator
        self.head = head

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = Boxcoder(reg_weights)

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1


    def create_proposal(self, anchor, objectness, pred_box_delta, image_shpae):
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']

        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx = objectness.topk(pre_nms_top_n)[1]
        score = objectness[top_n_idx]
        proposal = self.box_coder.decode(pred_box_delta[top_n_idx], anchor[top_n_idx])
        # print("@@@@proposal : ", proposal)
        proposal, score = process_box(proposal, score, image_shpae, self.min_size)
        # print("@@@ thresh : ", self.nms_thresh, type(self.nms_thresh))
        keep = nms(proposal, score, self.nms_thresh)
        proposal = proposal[keep]
        return proposal
        
    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)

        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        # print("pos nevgative idx : ", len(idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss

    def forward(self, feature, image_shape, target=None):
        if target is not None:
            anchor = self.anchor_generator(feature, image_shape)
            gt_box = target['boxes'].to(anchor.device)
            # print("Anchor : ", anchor.shape)
            # print("device check : ", gt_box.device, anchor.device)
            
            objectness, pred_bbox_delta = self.head(feature)
            objectness = objectness.permute(0, 2, 3, 1).flatten()
            pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

            proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)
            if self.training:
                objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
                return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}



