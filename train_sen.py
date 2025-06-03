import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import sys
import torch.nn.functional as F
sys.path.append('../CIRCOD/')
from data.data import *
from torch.utils.data import DataLoader
from models.sen import *
from eval.eval_sen import *
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

def Arguements():
    parser = argparse.ArgumentParser(description='Co-Saliency Inspired Referring Camouflaged Discovery')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay value')
    parser.add_argument('--gpu_main', default=0, help='a list of gpus')
    parser.add_argument('--num_worker', default=4, help='numbers of worker')
    parser.add_argument('--train_batch_size', type=int, default=8, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size')
    parser.add_argument('--image_size', default=384, type=int, help='size of image')
    parser.add_argument('--epoches', default=40, help='epoches')
    parser.add_argument('--task', help='task to perform')
    parser.add_argument('--train_dataset', help='train dataset directory')
    parser.add_argument('--snapshot', default=None, help='model loading directory')
    parser.add_argument('--test_dataset', help='test dataset directory')
    parser.add_argument('--checkpoint', help='model saving directory')
    args = parser.parse_args()
    return args

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1e-3) / (union - inter + 1e-3)
    return (wbce+wiou).mean()

def initializing_model(args):
    device_main = torch.device('cuda:{}'.format(int(args.gpu_main)))
    torch.set_num_threads(2)
    torch.cuda.set_device(device_main)
    net = SEN(device_main).to(device_main)
    if(args.snapshot):
        net.load_state_dict(torch.load(args.snapshot, map_location=device_main))
        print("snapshot loaded")
    train_dataset = Dataset_Generation(args.train_dataset, task=args.task, image_size=args.image_size, mode='train', count=2)
    test_cod_dataset = Dataset_Generation(args.test_dataset, task=args.task, image_size=args.image_size, mode='test', count=1)
    train_loader = DataLoader(train_dataset,batch_size=int(args.train_batch_size),num_workers=2,shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_cod_dataset,batch_size=int(args.test_batch_size),num_workers=2,shuffle=True, collate_fn=custom_collate)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)
    criterion = nn.MSELoss()
    parameters = {
        'device_main': device_main,
        'net': net,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
    }
    return parameters

def train_one_epoch(args, model_initialization):
    loader = tqdm(enumerate(model_initialization['train_loader']), desc="Batch Progress: ", total=len(model_initialization['train_loader']))
    prev_loss = float('inf')
    train_losses = []
    train_bin_losses=[]
    train_sal_losses=[]
    model_initialization['scheduler'].step(np.mean(train_losses))
    t_start = time.time()
    model_initialization['net'].train()
    for i, sample in loader:
        sample = autoencoder_process(sample, args.image_size)
        cod_img = torch.stack(sample['cod_img']).to(model_initialization['device_main'])
        sal_mask = torch.stack(sample['cod_mask']).to(model_initialization['device_main'])
        bin_mask = torch.stack(sample['bin_mask']).to(model_initialization['device_main'])
        pred_sal, _, preds = model_initialization['net'].forward(cod_img)
        loss_mse = model_initialization['criterion'](pred_sal, sal_mask)
        ce_loss = structure_loss(preds, bin_mask)
        loss = loss_mse + ce_loss
        model_initialization['optimizer'].zero_grad()
        loss.backward()
        model_initialization['optimizer'].step()
        train_losses.append(loss.data.cpu().numpy())
        train_bin_losses.append(ce_loss.data.cpu().numpy())
        train_sal_losses.append(loss_mse.data.cpu().numpy())
    delta = time.time() - t_start
    is_better_loss = np.mean(train_losses) < prev_loss
    if(is_better_loss):
        prev_loss = np.mean(train_losses)
        torch.save(model_initialization['net'].state_dict(), args.model_file)
    print("epoch{} iter {}/{} train loss: {} and time: {}".format(epoch,i, len(model_initialization['train_loader']), np.mean(train_losses), delta))

    print("testing......")
    metrics, _, eval_loss, _ = evaluate(model_initialization['net'], model_initialization['test_loader'], model_initialization['device_main'], autoencoder_process, args.image_size, args.test_batch_size, model_initialization['criterion'], structure_loss)
    print('metrics: {}, eval_loss: {}'.format(metrics, eval_loss))
    model_initialization['scheduler'].step(np.mean(train_losses))
    print("-----------------------------------------------------------------------------------------------")

if __name__=='__main__':
    args = Arguements()
    model_initialization = initializing_model(args)
    for epoch in tqdm(range(int(args.epoches)), desc="Epoch Progress: "):
        train_one_epoch(args, model_initialization)
