import torch

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # for p in optimizer.param_groups:
        # p['lr'] - 
    for i, (image, target) in enumerate(data_loader):

        num_iters = epoch * len(data_loader) + i
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        losses = model(image, target)
        total_loss = sum(losses.values())
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad
        print(f"{num_iters}: total_loss : {total_loss}")