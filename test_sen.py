import torch
import argparse
import sys

sys.path.append('../CIRCOD/')
from data.data import *
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.sen import *
from tqdm import tqdm
from metrics import *
torch.autograd.set_detect_anomaly(True)
import warnings
warnings.filterwarnings('ignore')

def Arguements():
    parser = argparse.ArgumentParser(description='Co-saliency inspired referring camouflaged discovery')
    parser.add_argument('--gpu_main', default=0, help='a list of gpus')
    parser.add_argument('--image_size', default=384, type=int, help='input image size')
    parser.add_argument('--dataset', help='test dataset directory')
    parser.add_argument('--task', help='task description')
    parser.add_argument('--snapshot', help='model saving directory')
    args = parser.parse_args()
    return args

def initializing_model(args):
    device_main = torch.device('cuda:{}'.format(int(args.gpu_main)))
    torch.set_num_threads(2)
    net = SEN().to(device_main)
    net.load_state_dict(torch.load(args.snapshot, map_location=device_main))
    test_dataset = Dataset_Generation(args.dataset, task=args.task, image_size=args.image_size, mode='test', count=1)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=4,shuffle=True, collate_fn=custom_collate)
    parameters = {
        'device_main': device_main,
        'net': net,
        'test_loader': test_loader,
        'image_size': args.image_size,
        'transform': T.ToPILImage()
    }
    return parameters

def testing(model_initialization):
    WFM = WeightedFmeasure()
    FM = Fmeasure()
    SM = Smeasure()
    EM = Emeasure()
    MA = MAE()
    model_initialization['net'].to(model_initialization['device_main']).eval()
    loader = tqdm(enumerate(model_initialization['test_loader']), desc="Batch Progress: ", total=len(model_initialization['test_loader']))
    for i, sample in loader:
        sample = autoencoder_process(sample, model_initialization['image_size'], mode='test')
        
        cod_img = torch.stack(sample['cod_img']).to(model_initialization['device_main'])
        gt = sample['bin_label_unsized'][0]
        _, pred_mask, _ = model_initialization['net'].forward(cod_img)
        
        gt = gt.squeeze().float().cpu().numpy()
        temp = Resize((max(gt.shape), max(gt.shape)))(pred_mask).squeeze()
        res = temp[0:gt.shape[0], 0:gt.shape[1]]
        res = torch.where(res >= 0.5, torch.tensor(1.).to(model_initialization['device_main']), torch.tensor(0.).to(model_initialization['device_main'])).data.cpu().numpy().squeeze()
        FM.step(pred=res*255, gt=gt*255)
        WFM.step(pred=res*255, gt=gt*255)
        SM.step(pred=res*255, gt=gt*255)
        EM.step(pred=res*255, gt=gt*255)
        MA.step(pred=res*255, gt=gt*255)   
    sm = SM.get_results()['sm'].round(3)
    adpem = EM.get_results()['em']['adp'].round(3)
    maxem = EM.get_results()['em']['curve'].max().round(3)
    meanem = EM.get_results()['em']['curve'].mean().round(3)
    wfm = WFM.get_results()['wfm'].round(3)
    meanFm= FM.get_results()['fm']["curve"].mean().round(3)
    mae = MA.get_results()['mae'].round(3)

    return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae, 'maxem': maxem, 'meanem': meanem, 'meanFm': meanFm}

if __name__=='__main__':
    args = Arguements()
    model_initialization = initializing_model(args)
    a = testing(model_initialization)
    print("test: ", a)