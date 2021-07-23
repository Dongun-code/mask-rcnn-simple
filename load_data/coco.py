import os
import sys
sys.path.append('..')
import torch
from PIL import Image
from pycocotools.coco import COCO
from config import Config as cfg
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
gc.collect()
torch.cuda.empty_cache()

class coco_set(torch.utils.data.Dataset):
    """
    input:
    path : COCO Dataset Path
    split : train, val or test

    return:
    img " [N, height, width]
    bbox : [N, [x1, y1, x2, y2]], [N, 4]
    category : [one-hot-encoding], [N, 91]
    masks : [N, height, width]
    """
    def __init__(self, path, split):
        ann = os.path.join(path, "annotations", f"instances_{split}2014.json")
        self.img_path = os.path.join(cfg.COCO_PATH, {'train' : 'train2014', 'val' : 'val2014'}[split])
        self.coco = COCO(ann)
        catIds = self.coco.getCatIds(catNms=['person', 'car','bus','bicyle', 'motorcycle'])
        self.imgIds = self.coco.getImgIds(catIds=catIds)
        self.cls_num = 91

    def __getitem__(self, idx):
        img_index = self.imgIds[idx]
        labelIds = self.coco.getAnnIds(imgIds=img_index)
        img = self.coco.loadImgs(img_index)[0]
        labels = self.coco.loadAnns(labelIds)

        img_file , height, width, image_id = img['file_name'], img['height'], img['width'], img['id']
        img_path = os.path.join(self.img_path,img_file)
        img = Image.open(img_path).convert('RGB')
        # cv_img = cv2.imread(img_path)

        boxes = []
        category = []
        masks = []

        if len(labels) > 0:
            for label in labels:
                if label['bbox'][2] < 1 or label['bbox'][3] < 1:
                    continue
                
                #   Need re-write 
                category_id = label['category_id']
                # id = np.zeros((self.cls_num,))
                # id[category_id] = 1
                category.append(category_id)

                bbox = label['bbox']
                # cv2.rectangle(cv_img, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])), (255, 0, 0),1)
                # plt.subplot(1, 2, 1)
                # plt.imshow(cv_img)
                boxes.append(bbox)

                mask = self.coco.annToMask(label)
                # plt.subplot(1, 2, 2)
                # plt.imshow(mask)
                # plt.show()
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
            # bbox = int(bbox)
            
            num_objs = len(labels)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            img = transforms.ToTensor()(img)
            img_shape =  img.shape[-2:]
            img_shape = torch.tensor(img_shape, dtype=torch.uint8)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            category = torch.tensor(category)
            masks = torch.stack(masks)
            image_id = torch.tensor([idx])
            # obj_ids = np.nique(masks)
            target = dict(boxes=boxes, labels=category, masks=masks, image_id=image_id, area=area, iscrowd=iscrowd, img_shape=img_shape)

        return img, target

    def convert_to_xyxy(self, box):
        #   box format (xmin, ymin, w, h)
        new_box = torch.zeros_like(box)
        new_box[:,0] = box[:, 0]
        new_box[:,1] = box[:, 1]
        new_box[:,2] = box[:, 0] + box[:, 2]
        new_box[:,3] = box[:, 1] + box[:, 3]

        return new_box

    def __len__(self):
        return len(self.imgIds)


if __name__ == "__main__":
    coco = coco_set(cfg.COCO_PATH, 'val')
    img, target = coco[1]
    # print(img.shape, category.shape, boxes.shape, masks.shape)
