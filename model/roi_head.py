from torch.functional import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.util import Matcher, BalancedPositiveNegativeSampler, roi_align
from model.box_ops import Boxcoder, box_iou, process_box, nms
from typing import List, Tuple
from torch import Tensor

def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    # print("test : ", class_logit, label)
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss


@torch.jit._script_if_tracing
def _onnx_paste_masks_in_image_loop(masks, boxes, im_h, im_w):
    res_append = torch.zeros(0, im_h, im_w)
    for i in range(masks.size(0)):
        mask_res = _onnx_paste_mask_in_image(masks[i][0], boxes[i], im_h, im_w)
        mask_res = mask_res.unsqueeze(0)
        res_append = torch.cat((res_append, mask_res))
    return res_append

def expand_boxes(boxes, scale):
    # type: (Tensor, float) -> Tensor
    if torchvision._is_tracing():
        return _onnx_expand_boxes(boxes, scale)
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

def expand_masks(mask, padding):
    # type: (Tensor, int) -> Tuple[Tensor, float]
    M = mask.shape[-1]
    if torch._C._get_tracing_state():  # could not import is_tracing(), not sure why
        scale = expand_masks_tracing_scale(M, padding)
    else:
        scale = float(M + 2 * padding) / M
    padded_mask = F.pad(mask, (padding,) * 4)
    return padded_mask, scale

def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    if torchvision._is_tracing():
        return _onnx_paste_masks_in_image_loop(masks, boxes,
                                               torch.scalar_tensor(im_h, dtype=torch.int64),
                                               torch.scalar_tensor(im_w, dtype=torch.int64))[:, None]
    res = [
        paste_mask_in_image(m[0], b, im_h, im_w)
        for m, b in zip(masks, boxes)
    ]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)[:, None]
    else:
        ret = masks.new_empty((0, 1, im_h, im_w))
    return ret


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

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = Boxcoder(reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = num_trhesh
        self.num_detections = num_detections
        self.min_size = 1

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes'].to(proposal.device)
        gt_label = target['labels'].to(proposal.device)
        proposal = torch.cat((proposal, gt_box))

        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))

        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        return proposal, matched_idx, label, regression_target
    

    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes =  class_logit.shape

        device = class_logit.deivce
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)

        boxes = []
        labels = []
        scores = []

        for i in range(1, num_classes):
            score, box_delta = pred_score[:, i], box_regression[:, i]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)

            box, score = process_box(box, score, image_shape, self.min_size)

            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, scores = box[keep], score[keep]
            label = torch.full((len(keep),), 1, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)
        # print("boxes boxes cat", boxes.shape, torch.cat(boxes).shape)
        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results


    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        # print("Box feature : ", box_feature.shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]

                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]

                '''
                ???
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''                
            if mask_proposal.shape[0] == 0:
                losses.update(dict(roi_mask_loss=torch.tensor(0)))
                return result, losses
        else:
            # print("result: ", result)
            mask_proposal = result['boxes']

            if mask_proposal.shape[0] == 0:
                result.update(dict(masks=torch.empty((0, 28, 28))))
                return result, losses

        mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
        mask_logit = self.mask_predictor(mask_feature)

        if self.training:
            gt_mask = target['masks']
            mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
            losses.update(dict(roi_mask_loss=mask_loss))
        else:
            label = result['labels']
            idx = torch.arange(label.shape[0], device=label.device)
            mask_logit = mask_logit[idx, label]

            mask_prob = mask_logit.sigmoid()
            result.update(dict(masks=mask_prob))

        return result, losses