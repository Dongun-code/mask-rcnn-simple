import torch
import math

from torch.nn.functional import threshold

class Boxcoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, referebce_box, proposal):
        """
        Encode a set of poroposals with respect to some
        reference boxes
        """

        width = proposal[:, 2] - proposal[:, 0]
        height = proposal[:, 3] - proposal[:, 1]
        ctr_x = proposal[:, 0] + 0.5 * width
        ctr_y = proposal[:, 1] + 0.5 * height

        gt_width = referebce_box[:, 2] - referebce_box[:, 0]
        gt_height = referebce_box[:, 3] - referebce_box[:, 1]
        gt_ctr_x = referebce_box[:, 0] + 0.5 * gt_width
        gt_ctr_y = referebce_box[:, 1] + 0.5 * gt_height

        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log(gt_width / width)
        dh = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta

    def decode(self, delta, box):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            delta (Tensor[N, 4]): encoded boxes.
            boxes (Tensor[N, 4]): reference boxes.
        """

        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        dw = torch.clamp(dw, max = self.bbox_xform_clip)
        dh = torch.clamp(dh, max = self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        pred_ctr_x = dx * width * ctr_x
        pred_ctr_y = dy * height * ctr_y
        pred_w = torch.exp(dw) * width
        pred_h = torch.exp(dh) * height

        xmin = pred_ctr_x - 0.5 * pred_w
        ymin = pred_ctr_y - 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h

        target = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return target

def process_box(box, score, image_shape, min_size):
    """
    Clip boxes in the image size and remove boxes which are too small
    """
    
    # box[:, [0, 2]] = box[:, [0,2]].clamp(0, image_shape[1])
    # box[:, [1, 3]] = box[:, [1,3]].clmap(0, image_shape[0])

    box[:, [0, 2]] = torch.clamp(box[:, [0,2]], min = 0, max = image_shape[1])
    box[:, [1, 3]] = torch.clamp(box[:, [1,3]], min = 0, max = image_shape[0])
    # print(box.shape)
    # print("@@@@@box : ", box)

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    # print("w, h : ", w, h)
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    # print("keep : ", keep)
    box, score = box[keep], score[keep]
    return box, score



def box_iou(box_a, box_b):
    """
    Arguments:
        boxe_a (Tensor[N, 4])
        boxe_b (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in box_a and box_b
    """
    # print("box device", box_a.device, box_b.device)
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)

    return inter / (area_a[:, None] + area_b - inter)


def nms(box, score, thresold):
    """
    Arguments:
        box: (Tensor[N, 4])
        score: (Tensor[N]) : score of the boxes.
        thresold (float): iou thresold

    Returns:
        keep (Tensor): indices of boxes filtered by NMS.
    """

    return torch.ops.torchvision.nms(box, score, thresold)
