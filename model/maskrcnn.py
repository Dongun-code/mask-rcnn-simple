from model.rpnetowrk import RPNHead, RPNetwork
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import backbone_factory
from model.util import AnchorGenerator
from model.pooler import ROIAlign
from model.transfrom import Transformer
from model.roi_head import ROIHeads
from collections import OrderedDict
from model.pooler import ROIAlign


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

        self.backbone = backbone_factory('resnet101', stage5 = True)
        out_channels = 2048
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
        print("ROI ALign shape" , box_roi_pool.output_size)
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = ROIHeads(
            box_roi_pool, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_num_sampeles, box_positive_fraction,
            box_reg_weights,
            box_score_thresh, box_nms_thresh, box_num_detection            
        )

        self.head.mask_roi_pool = ROIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        self.transformer = Transformer(
            min_size=800, max_size=1300,
            image_mean=[0.485, 0.456, 0.406],
            image_std = [0.299, 0.244, 0.225])        


    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]

        img = image.cuda()
        # print("im avlie")
        feature, C5 = self.backbone(img)

        proposal, rpn_losses = self.rpn(C5, image_shape, target)
        # print("proposal", proposal.shape)
        result, roi_losses = self.head(C5, proposal, image_shape, target)

        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postpreocess(result, image_shape, ori_image_shape)
            return result



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


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

# class resnet101_maskrcnn(nn.Module):
#     def __init__(self):
#         super().__init__()

