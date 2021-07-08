from torch.nn import parameter
from model.maskrcnn import MaskRCNN
from load_data.coco import coco_set
from config import Config as cfg
import torch
# from torchsummary import summary
import gc
gc.collect()
torch.cuda.empty_cache()


def get_gpu_prop(show=False):

    ngpus = torch.cuda.device_count()
    
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            "name": prop.name,
            "capability": [prop.major, prop.minor],
            "total_momory": round(prop.total_memory / 1073741824, 2), # unit GB
            "sm_count": prop.multi_processor_count
        })
       
    if show:
        print("cuda: {}".format(torch.cuda.is_available()))
        print("available GPU(s): {}".format(ngpus))
        for i, p in enumerate(properties):
            print("{}: {}".format(i, p))
    return properties


def main():
    device = torch.device("cuda")
    if device.type == "cuda": 
        get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    dataset_train = coco_set(cfg.COCO_PATH, 'train')
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    model = MaskRCNN('resnet101', 91).to(device)

    for image, target in dataset_train:
        model(image, target)

    # d_test = coco_set(cfg.COCO_PATH, 'val')

    num_classes = d_train.dataset.cls_num + 1
    
    # optimizer = torch.optim.SGD(mask.parameters(), lr=0.01, momentum=0.9)
    # print(optimizer)





if __name__ == "__main__":
    main()

