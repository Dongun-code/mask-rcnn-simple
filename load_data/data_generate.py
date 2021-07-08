import torch
from config import Config as cfg
import numpy as np
from model.module import generate_pyramid_anchors
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augment=True):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.
            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)
            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.
            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
        """
        #   batch item index
        self.b = 0
        self.image_index = -1
        #   Hmm..
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.augment = augment

        #   Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = generate_pyramid_anchors(cfg.RPN_ANCHOR_SCALES,
                                                cfg.RPN_ANCHOR_RATIOS,
                                                cfg.BACKBONE_SHAPES,
                                                cfg.BACKBONE_STRIDES,
                                                cfg.RPN_ANCHOR_STRIDE)
        
    def __getitem__(self, idx):
        #   Get bounding boxes and masks for image
        image_id = self.image_ids[idx]
        # image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
        #     load_image_gt

