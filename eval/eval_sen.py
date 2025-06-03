import torch
from tqdm import tqdm
import metrics as Measure
from data.utils import *
import numpy as np
torch.autograd.set_detect_anomaly(True)

def evaluate(net, test_loader, device_main, autoencoder_process, image_size, batch_size, criterion, structure_loss):
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    net.to(device_main).eval()
    eval_losses=[]
    eval_bin_losses=[]
    eval_sal_losses=[]
    loader = tqdm(enumerate(test_loader), desc="Batch Progress: ", total=len(test_loader))
    for i, sample in loader:
        sample = autoencoder_process(sample, image_size, mode='test')
        cod_img = torch.stack(sample['cod_img']).to(device_main)
        sal_mask = torch.stack(sample['cod_mask']).to(device_main)
        bin_mask = torch.stack(sample['bin_mask']).to(device_main)
        bin_label_unsized = sample['bin_label_unsized']
        pred_sal, pred_mask, preds = net.forward(cod_img)
        eval_lossy = criterion(pred_sal, sal_mask)
        ce_loss = structure_loss(preds, bin_mask)
        loss = eval_lossy + ce_loss
        res = torch.where(pred_mask >= 0.5, torch.tensor(1.).to(device_main), torch.tensor(0.).to(device_main))

        
        eval_losses.append(loss.data.cpu().numpy())
        eval_bin_losses.append(ce_loss.data.cpu().numpy())
        eval_sal_losses.append(eval_lossy.data.cpu().numpy())
        for i in range(len(cod_img)):
            gt = bin_label_unsized[i].squeeze().long().float().cpu().numpy()
            temp = Resize((max(gt.shape), max(gt.shape)))(res[i]).squeeze()
            temp = temp[0:gt.shape[0], 0:gt.shape[1]].data.cpu().numpy().squeeze()
            WFM.step(pred=temp*255, gt=gt*255)
            SM.step(pred=temp*255, gt=gt*255)
            EM.step(pred=temp*255, gt=gt*255)
            MAE.step(pred=temp*255, gt=gt*255)
        
    sm = SM.get_results()['sm'].round(3)
    adpem = EM.get_results()['em']['adp'].round(3)
    wfm = WFM.get_results()['wfm'].round(3)
    mae = MAE.get_results()['mae'].round(3)

    return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae}, np.mean(eval_bin_losses), np.mean(eval_losses), np.mean(eval_sal_losses)