from model.rpnetowrk import RPNHead, RPNetwork
import torch.nn as nn
import torch.functional as F
from model.backbone import backbone_factory
from model.util import AnchorGenerator
from model.pooler import ROIAlign
from model.transfrom import Transformer

class MaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes,
                 #  RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 #  ROI heads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_sampeles=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detection=100):

        super().__init__()
        # self.backbone = backbone
        # self.channels = backbone.out_channels
        out_channels = 256
        self.backbone = backbone_factory('resnet101', stage5 = True)

        #   RPN
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        print("rpn_anchor_generator : ", rpn_anchor_generator)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RPNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_num_samples, rpn_positive_fraction,
            rpn_reg_weights,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #   ROI Align
        box_roi_pool = ROIAlign(output_size=(7, 7), sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        # self.head = ROIHeads(
            
        # )
        # self.head.mask_roi_pool = ROIAlign(output_size=(14, 14), sampling_ratio=2)
        self.transformer = Transformer(
            min_size=800, max_size=1300,
            image_mean=[0.485, 0.456, 0.406],
            image_std = [0.299, 0.244, 0.225])        


    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        image, target = self.transformer(image, target)
        # print(image)
        # print(image.shape)
        # image_shape = image.shape[-2:]
        img = image.cuda()
        feature = self.backbone(img)

        # proposal, rpn_losses = self.rpn(feature, image_shape, target)
        # result, roi_losses = self.head


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta

