import math
from numpy.lib.twodim_base import mask_indices
import torch
from torch import nn, Tensor
from torch._C import device
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import torchvision
from model.image_list import ImageList
import torch.nn as nn
from model.roi_head import paste_masks_in_image

@torch.jit.unused
def _get_shape_onnx(image):
    # type: (Tensor) -> Tensor
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]

@torch.jit.unused
def _fake_cast_onnx(v):
    # type: (Tensor) -> float
    # ONNX requires a tensor but here we fake its type for JIT.
    return v

def _resize_image_and_masks(image: Tensor, self_min_size: float, self_max_size: float,
                            target: Optional[Dict[str, Tensor]] = None,
                            fixed_size: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        img_shpae = _get_shape_onnx(image)
    else:
        img_shpae = torch.tensor(image.shape[-2:])
    
    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None

    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(img_shpae).to(dtype=torch.float32)
        max_size = torch.max(img_shpae).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True
    
    image = torch.nn.functional.interpolate(image[None], size=size, scale_factor=scale_factor, mode='bilinear',
                                            recompute_scale_factor=recompute_scale_factor, align_corners=False)[0]

    if target is None:
        return image, target
    
    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor,
                                                recompute_scale_factor=recompute_scale_factor)[:, 0].byte()

        target["masks"] = mask

    return image, target


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std, size_divisible=32, fixed_size=None):
        super(Transformer, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)        
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible= size_divisible
        self.fixed_size = fixed_size


    def normalized(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )            

        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]

    def resize_boxes(self, boxes, original_size, new_size):
        # type: (Tensor, List[int], List[int]) -> Tensor
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]

        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height

        return torch.stack((xmin, ymin, xmax, ymax), dim=1)


    def resize(self,
               image: Tensor,
               target: Optional[Dict[str, Tensor]] = None,
               ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target


        bbox = target["boxes"]
   
        bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox   

        return image, target

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        print("the list : ", the_list)
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                print("in : ", maxes[index], item)
                maxes[index] = max(maxes[index], item)
        return maxes

    def batched_image(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)
        
        max_size = self.max_by_axis([list(img.shape) for img in images])
        print("@@ max_size : ", max_size)
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        print("@@@ max_size : ", max_size)
        batch_shape = [len(images)] + max_size
        batched_imgs  = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

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

    def forward(self,
                images,          # type : List [Tensor]
                targets= None    # type : Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy : List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"image is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            
            image = self.normalized(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batched_image(images, size_divisible=self.size_divisible)

        image_size_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_size_list.append((image_size[0], image_size[1]))
        
        image_list = ImageList(images, image_size_list)
        return image_list, targets

            





    