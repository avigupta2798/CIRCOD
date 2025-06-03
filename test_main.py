import torch
from tqdm import tqdm
import argparse
import sys

sys.path.append('../CIRCOD/')
from metrics import *
import numpy as np
from torch.utils.data import DataLoader
from data.utils import *
from data.data import *
import torchvision.transforms as T
import torch.nn.functional as F
from models.circod import *
torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings('ignore')

def Arguements():
    parser = argparse.ArgumentParser(description='Conditional Referring Object detection')
    parser.add_argument('--gpu', default=0, help='a list of gpus')
    parser.add_argument('--batch_size', type=int, default=1, help='bacth size')
    parser.add_argument('--image_size', default=512, type=int, help='validation label directory')
    parser.add_argument('--cod_test_dataset', help='model saving directory')
    parser.add_argument('--search_dataset', help='model saving directory')
    parser.add_argument('--sen_path', help='model saving directory')
    parser.add_argument('--snapshot', help='model saving directory')
    parser.add_argument('--task', help='task to perform')
    args = parser.parse_args()
    return args

def best_model(args, device):
    net = CIRCOD(device, args.sen_path).to(device)
    print('step one done')
    net.load_state_dict(torch.load(args.snapshot, map_location=device))
    return net

def create_loader():
    test_cod_dataset = Dataset_Generation(args.cod_test_dataset, args.search_dataset, task=args.task, image_size=args.image_size, mode='test', count=1)
    test_loader = DataLoader(test_cod_dataset,batch_size=int(args.batch_size),num_workers=4,shuffle=True)
    return test_loader

transform = T.ToPILImage()

def test(net, test_loader, device_main, image_size):
    
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    FM = Fmeasure()

    net.to(device_main).eval()
    test_pred_decision = []
    test_true_decision = []
    loader = tqdm(enumerate(test_loader), desc="Batch Progress: ", total=len(test_loader))
    for i, sample in loader:
        sample = pair_conversion(sample, image_size, mode='eval')
        cod_img = torch.stack(sample['cod_img']).to(device_main)
        cod_mask_orig = torch.stack(sample['cod_mask_orig']).to(device_main)
        si_img = torch.stack(sample['si_img']).to(device_main)
        si_mask = torch.stack(sample['si_mask']).to(device_main)
        cod_label = torch.tensor(sample['cod_label'])
        si_label = torch.tensor(sample['si_label'])
        output2, _, decision, estimated_flow, _ = net.forward(cod_img, si_img)
        si_mask = remap_using_flow_fields((si_mask*255).byte().permute(0,2,3,1).detach().cpu().numpy(), estimated_flow[:,0].detach().cpu().numpy(),estimated_flow[:,1].detach().cpu().numpy())
        si_mask = torch.stack(si_mask).to(device_main)
        decision = torch.where(decision >= 0.5, torch.tensor(1).to(device_main), torch.tensor(0).to(device_main))
        if(cod_label==si_label):
            truth = torch.tensor(1).float().to(device_main)
        else:
            truth = torch.tensor(0).float().to(device_main)
        test_pred_decision.append(decision.cpu().detach())
        test_true_decision.append(truth.cpu().detach())
        cod_mask_orig = cod_mask_orig*truth
        output2 = torch.where(output2.sigmoid() >= 0.5, torch.tensor(1).to(device_main), torch.tensor(0).to(device_main))
        gt = cod_mask_orig.squeeze().long().float().cpu().numpy()
        res = Resize((max(gt.shape), max(gt.shape)))(output2).squeeze()
        res = res[0:gt.shape[0], 0:gt.shape[1]]
        
        res = res.data.cpu().numpy().squeeze()
        if(cod_label==si_label):
            WFM.step(pred=res*255, gt=gt*255)
            SM.step(pred=res*255, gt=gt*255)
            EM.step(pred=res*255, gt=gt*255)
            FM.step(pred=res*255, gt=gt*255)
            M.step(pred=res*255, gt=gt*255)
        else:
            WFM.step(pred=(1-res)*255, gt=(1-(gt*0))*255)
            SM.step(pred=(1-res)*255, gt=(1-(gt*0))*255)
            EM.step(pred=(1-res)*255, gt=(1-(gt*0))*255)
            FM.step(pred=(1-res)*255, gt=(1-(gt*0))*255)
            M.step(pred=(1-res)*255, gt=(1-gt)*255)    
    sm = SM.get_results()['sm'].round(3)
    adpem = EM.get_results()['em']['adp'].round(3)
    maxem = EM.get_results()['em']['curve'].max().round(3)
    meanem = EM.get_results()['em']['curve'].mean().round(3)
    wfm = WFM.get_results()['wfm'].round(3)
    meanFm= FM.get_results()['fm']["curve"].mean().round(3)
    mae = M.get_results()['mae'].round(3)

    test_pred_decision = torch.stack(test_pred_decision)
    test_true_decision = torch.stack(test_true_decision)
    test_pred_decision = (test_pred_decision >= 0.5).int()
    final_preds = (np.count_nonzero(test_pred_decision.squeeze()==test_true_decision))/len(test_true_decision)*100
    return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae, 'maxem': maxem, 'meanem': meanem, 'meanFm': meanFm}, final_preds

if __name__=='__main__':
    args = Arguements()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    model = best_model(args, device)
    test_loader = create_loader()
    metrics, final_preds = test(model, test_loader, device, args.image_size)
    print(metrics)
    print()
    print(final_preds)