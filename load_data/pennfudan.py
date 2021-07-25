import os 
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #   load all image files, sorting them to
        #   ensure that thy are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.path.join(roort, "PedMasks")))


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        #  0 being background
        mask = Image.open(mask_path)
        mask = np.array(mask)
        #   instance are encoded as different colors
        obj_ids = np.unique(mask)
        #   first id is background
        obj_ids = obj_ids[1:]

        #   split the color-encoded mask into a  set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        print(mask.shape)
        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        #   convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        #   there is only one class
        labels = torch.ones((num_objs,), dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[: 2], - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

# if __name__ == "__main__":
#     path = "/home/user/segment_data/PennFudanPed"
#     transform = ''
#     PennFudanDataset(path,)

