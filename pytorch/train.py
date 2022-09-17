import torch
from utils.datasets import YoloDataset
from utils.preprocessing import train_transform, test_transform
from config import (
        EPOCHS, LEARNING_RATE, 
        )

def create_dataloader(path, img_size, multiscale, transform, batch_size, shuffle):
    dataset = YoloDataset(path, img_size, multiscale, transform)
    dataloader = DataLoader(
            dataset, batch_size,
            shuffle = shuffle,
            pin_memory = True,
            collate_fn = dataset.collate_fn)
    return dataloader

def trainer(model, dataloader, device):
    for step in images, labels in tqdm(dataloader, decs = 'Train'):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        pred = model(images)
        loss, loss_components = yolo_loss_fn(pred, labels, model)

    return loss, loss_components

def tester(model, dataloader, device):
    for step in images, labels in tqdm(dataloader, desc = 'Val'):
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        pred = model(images)
        loss, loss_components - yolo_loss_fn(pred, labels, model)

    return pred, loss_components

def log(loss_components):
    pass

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader = create_dataloader(
            TRAIN_FILES, 416, True, train_transform, BATCH_SIZE, True)

    model = YOLOv3(9, 3)
    
    optimizer = optim.AdamW(
            model.parameter(), LEARNING_RATE,)

    len_batch = len(train_dataloader)
    for epcoh in range(1, EPCOHS + 1):
        loss, loss_components = trainer(model, train_dataloader, device)
        loss.backward()
        
        for g in optimizer.param_groups:
            g['lr'] = new_lr

        optimizer.step()
        optimizer.zero_grad(True)

        log(loss_components)

        pred, loss_components = tester(model, test_dataloader, device)
        pred = nms(pred, CONF_THRESH, IOU_THRESH)
        mAP = compute_ap(pred, IOU_THRESH)

if __name__ == '__main__':
    run()
