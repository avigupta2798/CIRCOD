import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import sys
sys.path.append('../CIRCOD/')

from data.data import *
from torch.utils.data import DataLoader
from models.circod import *
from eval.eval_main import *
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
    parser.add_argument('--task', help='task to perform')
    parser.add_argument('--image_size', default=384, type=int, help='size of input image')
    parser.add_argument('--epoches', default=40, help='epoches')
    parser.add_argument('--snapshot', default=None, help='model loading directory')
    parser.add_argument('--cod_train_dataset', help='train dataset directory')
    parser.add_argument('--cod_test_dataset', help='test dataset directory')
    parser.add_argument('--search_dataset', help='search dataset directory')
    parser.add_argument('--sen_path', help='SEN path')
    parser.add_argument('--checkpoint', help='model saving directory')
    args = parser.parse_args()
    return args

def custom_collate(data):
    return data

def initializing_model(args):
    device_main = torch.device('cuda')
    torch.set_num_threads(2)
    net = CIRCOD(device_main, args.sen_path).to(device_main)
    if(args.snapshot):
        net.load_state_dict(torch.load(args.snapshot, map_location=device_main))
    net.sen.load_state_dict(torch.load(args.sen_path, map_location=device_main))
    train_cod_dataset = Dataset_Generation(args.cod_train_dataset, args.search_dataset, task=args.task, image_size=args.image_size, mode='train', count=2)
    test_cod_dataset = Dataset_Generation(args.cod_test_dataset, args.search_dataset, task=args.task, image_size=args.image_size, mode='test', count=1)
    print(len(train_cod_dataset))
    print(len(test_cod_dataset))
    train_loader = DataLoader(train_cod_dataset,batch_size=int(args.train_batch_size),num_workers=4,shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_cod_dataset,batch_size=int(args.test_batch_size),num_workers=4,shuffle=True, collate_fn=custom_collate)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)
    criterion_bce = nn.BCELoss()
    parameters = {
        'device_main': device_main,
        'net': net,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion_bce': criterion_bce,
    }
    return parameters

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1e-3) / (union - inter + 1e-3)
    return wbce, wiou

def train_one_epoch(args, model_initialization):
    loader = tqdm(enumerate(model_initialization['train_loader']), desc="Batch Progress: ", total=len(model_initialization['train_loader']))
    prev_loss = float('inf')
    losses = []
    decision_losses = []
    mask_losses = []
    t_start = time.time()
    model_initialization['net'].train()
    train_pred_decision = []
    train_true_decision = []
    for i, sample in loader:
        sample = pair_conversion(sample, args.image_size)
        cod_img = torch.stack(sample['cod_img']).to(model_initialization['device_main'])
        cod_mask = torch.stack(sample['cod_mask']).to(model_initialization['device_main'])
        si_img = torch.stack(sample['si_img']).to(model_initialization['device_main'])
        si_mask = torch.stack(sample['si_mask']).to(model_initialization['device_main'])
        cod_label = torch.tensor(sample['cod_label'])
        si_label = torch.tensor(sample['si_label'])
        gt_decision = torch.where(cod_label == si_label, 1., 0.).to(model_initialization['device_main'])
        output2, output_si, decision, estimated_flow, y = model_initialization['net'].forward(cod_img, si_img)
        si_mask = remap_using_flow_fields((si_mask*255).byte().permute(0,2,3,1).detach().cpu().numpy(), estimated_flow[:,0].detach().cpu().numpy(),estimated_flow[:,1].detach().cpu().numpy())
        si_mask = torch.stack(si_mask).to(model_initialization['device_main'])
        
        train_pred_decision.extend(decision.squeeze().cpu().detach())
        train_true_decision.extend(gt_decision.squeeze().cpu().detach())
        
        bce, iou = structure_loss(output2*decision.unsqueeze(1).unsqueeze(1), cod_mask*gt_decision.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        bce_si, iou_si = structure_loss(output_si*decision.unsqueeze(1).unsqueeze(1), si_mask*gt_decision.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        
        loss_decision = model_initialization['criterion_bce'](decision, gt_decision.unsqueeze(1))
        cod_mask_loss = torch.mean(bce + (gt_decision.unsqueeze(1) * iou))
        si_mask_loss = torch.mean(bce_si + (gt_decision.unsqueeze(1) * iou_si))
        
        loss = loss_decision + cod_mask_loss + si_mask_loss
        
        model_initialization['optimizer'].zero_grad()
        loss.backward()
        model_initialization['optimizer'].step()
        losses.append(loss.data.cpu().numpy())
        mask_losses.append(cod_mask_loss.data.cpu().numpy())
        decision_losses.append(loss_decision.data.cpu().numpy())
    train_pred_decision = torch.stack(train_pred_decision)
    train_true_decision = torch.stack(train_true_decision)
    threshold = 0.5
    train_pred_decision = (train_pred_decision >= threshold).int()
    train_decision_accuracy = (train_pred_decision == train_true_decision).float().mean().item() * 100.0

    delta = time.time() - t_start
    is_better_loss = np.mean(losses) < prev_loss
    if(is_better_loss):
        prev_loss = np.mean(losses)
        torch.save(model_initialization['net'].state_dict(), args.checkpoint)
    print("epoch{} iter {}/{} train loss: {} and time: {}".format(epoch,i, len(model_initialization['train_loader']), np.mean(losses), delta))

    print("testing......")
    metrics, eval_loss, eval_decision_accuracy = evaluate(model_initialization['net'], model_initialization['test_loader'], model_initialization['device_main'], pair_conversion, structure_loss, args.image_size, args.test_batch_size, model_initialization['criterion_bce'])
    
    print('metrics(sm, adpem, wfm, mae): {}, eval_loss: {}, eval_decision_accuracy: {}, train_decision_accuracy: {}'.format(metrics, eval_loss, eval_decision_accuracy, train_decision_accuracy))
    
    model_initialization['scheduler'].step(np.mean(losses))
    print("-----------------------------------------------------------------------------------------------")

if __name__=='__main__':
    args = Arguements()
    model_initialization = initializing_model(args)
    for epoch in tqdm(range(int(args.epoches)), desc="Epoch Progress: "):
        train_one_epoch(args, model_initialization)