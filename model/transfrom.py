import math
from numpy.lib.twodim_base import mask_indices
import torch
from torch._C import device
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Transformer:
    def __init__(self, min_size, max_size, image_mean, image_std):
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def __call__(self, image, target):
        self.target = target
        image = self.normalized(image)
        image, target = self.resize(image, target)
        # print("After resize : ", image.shape)
        image = self.batched_image(image)

        return image, target

    def normalized(self, image):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # print("@@@@@@@@2image", image.shape)

        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # print('@@@@@Image :', image.shape, image.shape[-2:])
        ori_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        # print("max, min", max_size, min_size)

        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        # print("@@@@@@@scale factor: ",self.min_size / min_size, self.max_size / max_size)
        size = [round(s * scale_factor) for s in ori_image_shape]
        # print("@@@@@@@@Size : ", size)
        image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]
        # print("Inter : ", image.shape)
        if target is None:
            return image, target
        
        box = target['boxes']
        # print("Box" , box)
        # print(image.shape[-1] , ori_image_shape[1])
        box[:,[0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
        target['boxes'] = box

        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), size=size)[0].byte()
            # print(mask.shape)
            target['masks'] = mask

        return image, target


    def batched_image(self, image, stride=32):
        size = image.shape[-2:]
        max_size = tuple(math.ceil(s / stride) * stride for s in size)

        batch_shape = (image.shape[-3],) + max_size
        batched_img = image.new_full(batch_shape, 0)
        batched_img[:, :image.shape[-2], :image.shape[-1]] = image

        return batched_img[None]

    def postpreocess(self, result, image_shape, ori_image_shape):
        box = result['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0]
        result['boxes'] = box

        if 'masks' in result:
            mask = result['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            result['mask'] = mask
        return result

    