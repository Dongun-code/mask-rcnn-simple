from torch.nn import parameter
from model.maskrcnn import MaskRCNN
from load_data.coco import coco_set
from config import Config as cfg
import torch
from train_one import train_one_epoch
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

    dataset_train = coco_set(cfg.COCO_PATH, 'val')
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    model = MaskRCNN('resnet101', 92).to(device)
    momentum = 0.9
    # lr_temp = 0.02 * 1 / 16
    lr_temp = 0.0001
    weight_decay_ = 0.0001
    device = 'cuda'
    params = [p for p in model.parameters() if p.requires_grad]
    # print("parameter : ", params)
    optimizer = torch.optim.SGD(
        params, lr=lr_temp, momentum=momentum, weight_decay= weight_decay_
    )
    model.train()
    start_epoch = 0
    end_epoch = 5

    for epoch in range(start_epoch, end_epoch):
        print((f"@@@[Epoch] : {epoch}"))
        train_one_epoch(model, optimizer, d_train, device, epoch)
    # for image, target in dataset_train:
    #     losses = model(image, target)
    #     print("RPN loss, ROI loss : ", losses)
    # d_test = coco_set(cfg.COCO_PATH, 'val')

    num_classes = d_train.dataset.cls_num + 1
    
    # optimizer = torch.optim.SGD(mask.parameters(), lr=0.01, momentum=0.9)
    # print(optimizer)





if __name__ == "__main__":
    main()

