import math
import torch
from torch import nn, Tensor
from typing import List, Optional
from model.image_list import ImageList

class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and image sizes.

    The module support computing anchors at multiple sizes and aspect ratios per feature map.
    This module assumes aspect ratio = height / width for each anchor

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i]
    anchors per spatial location for feature map i


    Args:
        sizes (Tuple[Tuple[int]])
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors" : List[torch.Tensor]
    }

    def __init__(
        self, 
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2),),
    )
    super(AnchorGenerator, self).__intit__()

    if not isinstance(sizes[0], (list, tuple)):
        sizes = tuple((s,) for s in sizes)
    if not isinstance(aspect_ratios[0], (list, tuple)):
        aspect_ratios = (aspect_ratios,) * len(sizes)

    # assert len(sizes) == len(aspect_ratios)
