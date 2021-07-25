from torch.nn import parameter
from model.maskrcnn import MaskRCNN
from load_data.coco import coco_set
from config import Config as cfg
from model.detection.engine import train_one_epoch, evaluate
import torch
# from train_one import train_one_epoch
# from torchsummary import summary
import model.detection.utils as utils
import gc
import argparse
import time

gc.collect()
torch.cuda.empty_cache()
# from torchvision.references.detection import engine

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
    dataset_val = coco_set(cfg.COCO_PATH, 'val')
    indices = torch.randperm(len(dataset_train)).tolist()
    # d_train = torch.utils.data.Subset(dataset_train, indices)
    # d_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, 
    #                                         shuffle=True, num_workers=0)
    d_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, 
                                            shuffle=True, num_workers=0,
                                            collate_fn=utils.collate_fn)  
    d_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, 
                                            shuffle=False, num_workers=0,
                                            collate_fn=utils.collate_fn)                                                                                         
    # d_val = torch.utils.data.Subset()
    # print(d_train)

    model = MaskRCNN('resnet101', 92).to(device)
    momentum = 0.9
    # lr_temp = 0.02 * 1 / 16
    lr_temp = 0.005
    weight_decay_ = 0.0005
    device = 'cuda'
    params = [p for p in model.parameters() if p.requires_grad]
    # print("parameter : ", params)
    optimizer = torch.optim.SGD(
        params, lr=lr_temp, momentum=momentum, weight_decay= weight_decay_
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    # model.train()
    start_epoch = 0
    end_epoch = 1

    for epoch in range(start_epoch, end_epoch):
        
        tm_1 = time.time()
        print((f"@@@[Epoch] : {epoch + 1}"))
        # train_one_epoch(model, optimizer, d_train, device, epoch)
        train_one_epoch(model, optimizer, d_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        # evaluate(model, d_val, device=device)




    num_classes = d_train.dataset.cls_num + 1
    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--momentum", type=float, default=0.9)
    # parser.add_argument("--weight-decay", type=float, default=0.0001)
    # args = parser.parse_args()

    main()

