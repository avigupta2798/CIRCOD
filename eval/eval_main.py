import torch
from tqdm import tqdm
import metrics as Measure
import numpy as np
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from data.utils import *
from sklearn.metrics import confusion_matrix

def evaluate(net, test_loader, device_main, pair_conversion, structure_loss, image_size, batch_size, criterion_bce):
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    net.to(device_main).eval()
    eval_losses=[]
    eval_decision_losses = []
    eval_mask_losses = []
    test_pred_decision = []
    test_true_decision = []
    loader = tqdm(enumerate(test_loader), desc="Batch Progress: ", total=len(test_loader))
    for i, sample in loader:
        sample = pair_conversion(sample, image_size, mode='test')
        cod_img = torch.stack(sample['cod_img']).to(device_main)
        cod_mask = torch.stack(sample['cod_mask']).to(device_main)
        si_img = torch.stack(sample['si_img']).to(device_main)
        si_mask = torch.stack(sample['si_mask']).to(device_main)
        cod_label = torch.tensor(sample['cod_label'])
        si_label = torch.tensor(sample['si_label'])
        output2, output_si, decision, estimated_flow, _ = net.forward(cod_img, si_img)

        si_mask = remap_using_flow_fields((si_mask*255).byte().permute(0,2,3,1).detach().cpu().numpy(), estimated_flow[:,0].detach().cpu().numpy(),estimated_flow[:,1].detach().cpu().numpy())
        si_mask = torch.stack(si_mask).to(device_main)
        gt_decision = torch.where(cod_label == si_label, 1., 0.).to(device_main)
        test_pred_decision.extend(decision.squeeze().cpu().detach())
        test_true_decision.extend(gt_decision.squeeze().cpu().detach())
        
        bce, iou = structure_loss(output2*decision.unsqueeze(1).unsqueeze(1), cod_mask*gt_decision.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        bce_si, iou_si = structure_loss(output_si*decision.unsqueeze(1).unsqueeze(1), si_mask*gt_decision.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        
        loss_decision = criterion_bce(decision, gt_decision.unsqueeze(1))
        cod_mask_loss = torch.mean(bce + (gt_decision.unsqueeze(1) * iou))
        si_mask_loss = torch.mean(bce_si + (gt_decision.unsqueeze(1) * iou_si))
        
        loss=loss_decision + cod_mask_loss + si_mask_loss
        
        output2 = torch.where(output2.sigmoid() >= 0.5, torch.tensor(1.).to(device_main), torch.tensor(0.).to(device_main))
        res = output2.data.cpu().numpy().squeeze()
        decision = torch.where(decision >= 0.5, torch.tensor(1).to(device_main), torch.tensor(0).to(device_main))
        res = res*(decision.unsqueeze(1)).cpu().numpy()
        eval_losses.append(loss.data.cpu().numpy())
        eval_mask_losses.append(cod_mask_loss.data.cpu().numpy())
        eval_decision_losses.append(loss_decision.data.cpu().numpy())
        for m in range(len(cod_label)):
            if(cod_label[m]==si_label[m]):
                gt = cod_mask[m].squeeze().long().float().cpu().numpy()*255
            else:
                gt = cod_mask[m].squeeze().long().float().cpu().numpy()*0
                gt = (1-gt)*255
            WFM.step(pred=res[m]*255, gt=gt)
            SM.step(pred=res[m]*255, gt=gt)
            EM.step(pred=res[m]*255, gt=gt)
            MAE.step(pred=res[m]*255, gt=gt)
    test_pred_decision = torch.stack(test_pred_decision)
    test_true_decision = torch.stack(test_true_decision)
    threshold = 0.5

    test_pred_decision = (test_pred_decision >= threshold).int()
    test_decision_accuracy = (test_pred_decision == test_true_decision).float().mean().item() * 100.0

    sm = SM.get_results()['sm'].round(3)
    adpem = EM.get_results()['em']['adp'].round(3)
    wfm = WFM.get_results()['wfm'].round(3)
    mae = MAE.get_results()['mae'].round(3)
    
    return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae}, np.mean(eval_losses), test_decision_accuracy