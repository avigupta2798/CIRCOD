import torch
import torch.nn as nn
from models.pvtv2 import pvt_v2_b2
from models.sen import *
from models.models_compared import GLU_Net
from data.utils import *

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class LFM(nn.Module):
    def __init__(self):
        super(LFM, self).__init__()
        self.upsample_4_0 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_2_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

    def forward(self, x1, x2, x3, x4):
        x4_t_up = self.upsample_4_0(x4)
        x43_cat = torch.cat((x4_t_up, x3), dim=1)
        x43_cat_up = self.upsample_2_2(x43_cat)
        x432_cat = torch.cat((x43_cat_up, x2), dim=1)
        x432_cat_up = self.upsample_2_3(x432_cat)
        x4321_cat = torch.cat((x432_cat_up, x1), dim=1)
        return x4321_cat

class RCOS(nn.Module):
    def __init__(self, num_in=128, dropout=0.1):
        super(RCOS, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([num_in])).cuda()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sal, si):
        query_self = si
        key_self = si
        value_self = si
        query = si
        key = sal
        value = sal
        energy_self = (torch.matmul(query_self,key_self.permute(0, 1, 3, 2)))/self.scale
        attention_self = torch.softmax(energy_self, dim = 1)
        x_self = torch.matmul(self.dropout(attention_self), value_self)
        query = x_self
        key = sal
        value = sal
        energy = (torch.matmul(query,key.permute(0, 1, 3, 2)))/self.scale
        attention = torch.softmax(energy, dim = 1)
        x = torch.matmul(self.dropout(attention), value)
        out = sal + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.softmax(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SSP(nn.Module):
    def __init__(self):
        super(SSP, self).__init__()
        self.dc_linear = nn.Sequential(nn.Linear(128, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
        self.avgpool_x = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_y = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x, y):
        x = self.avgpool_x(x).squeeze()
        y = self.avgpool_y(y).squeeze()
        metric = self.dc_linear(torch.abs(x-y))
        return metric

class CIRCOD(nn.Module):
    def __init__(self, device, path, channel=64):
        super(CIRCOD, self).__init__()
        self.sen = SEN(device=device)
        sen_path = path
        self.sen.load_state_dict(torch.load(sen_path, map_location=device))
        print("sen loaded")
        for param in  self.sen.parameters():
            param.requires_grad = False
        self.backbone = pvt_v2_b2()
        path = './pre_trained_models/pvt_v2_b2.pth'
        self.network = GLU_Net(path_pre_trained_models='./pre_trained_models/',
                      model_type='DPED_CityScape_ADE',
                      consensus_network=False,
                      cyclic_consistency=True,
                      iterative_refinement=True,
                      apply_flipping_condition=False)
        self.Translayer2_0 = BasicConv2d(64, channel, 3, padding=1)
        self.Translayer2_1 = BasicConv2d(128, channel, 3, padding=1)
        self.Translayer3_1 = BasicConv2d(320, channel, 3, padding=1)
        self.Translayer4_1 = BasicConv2d(512, channel, 3, padding=1)
        save_model = torch.load(path, map_location=device)
        self.device= device
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.LFM = LFM()
        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()
        self.RCOS = RCOS()
        self.down05 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.ssp = SSP()
        self.upsample_cod_1 = nn.ConvTranspose2d(in_channels=128, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.upsample_cod_2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.upsample_si_1 = nn.ConvTranspose2d(in_channels=128, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.upsample_si_2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x, y):
        x_sal, x_mask, preds = self.sen(x)
        estimated_flow = self.network.estimate_flow((y*255).byte(), (x_sal*255).byte(), self.device, mode='channel_first')
        y = remap_using_flow_fields((y*255).byte().permute(0,2,3,1).detach().cpu().numpy(), estimated_flow[:,0].detach().cpu().numpy(),estimated_flow[:,1].detach().cpu().numpy())
        y = torch.stack(y).to(self.device)
        pvt_x = self.backbone(x_sal)
        pvt_y = self.backbone(y)
        
        x1 = pvt_x[0]
        x2 = pvt_x[1]
        x3 = pvt_x[2]
        x4 = pvt_x[3]
        
        x4_t = self.Translayer4_1(x4)
        x3_t = self.Translayer3_1(x3)
        x2_t = self.Translayer2_1(x2)
        x1_t = self.Translayer2_0(x1)
        
        y1 = pvt_y[0]
        y2 = pvt_y[1]
        y3 = pvt_y[2]
        y4 = pvt_y[3]
        
        y4_t = self.Translayer4_1(y4)
        y3_t = self.Translayer3_1(y3)
        y2_t = self.Translayer2_1(y2)
        y1_t = self.Translayer2_0(y1)
        
        lfm_feature_x = self.LFM(x1_t, x2_t, x3_t, x4_t)
        lfm_feature_y = self.LFM(y1_t, y2_t, y3_t, y4_t)

        ca_attended_x = self.ca(lfm_feature_x)
        ca_attended_y = self.ca(lfm_feature_y)
        sa_attended_x = self.sa(lfm_feature_x)
        sa_attended_y = self.sa(lfm_feature_y)

        ca_min = torch.min(ca_attended_x, ca_attended_y)
        ca_max = torch.max(ca_attended_x, ca_attended_y)
        sa_min = torch.min(sa_attended_x, sa_attended_y)
        sa_max = torch.max(sa_attended_x, sa_attended_y)
        
        ca_attention = ca_min/(ca_max+1e-5)
        sa_attention = sa_min/(sa_max+1e-5)

        lfm_feature_x = ca_attention * lfm_feature_x
        lfm_feature_x = sa_attention * lfm_feature_x
        
        lfm_feature_y = ca_attention * lfm_feature_y
        lfm_feature_y = sa_attention * lfm_feature_y
        
        lfm_feature_x_decision = self.down05(lfm_feature_x)
        lfm_feature_y_decision = self.down05(lfm_feature_y)
        
        sam_feature = self.RCOS(lfm_feature_x, lfm_feature_y)

        prediction_cod = self.upsample_cod_1(sam_feature)
        prediction_si = self.upsample_si_1(lfm_feature_y)

        prediction_cod = self.upsample_cod_2(prediction_cod)
        prediction_si = self.upsample_si_2(prediction_si)
        decision = self.ssp(lfm_feature_x_decision, lfm_feature_y_decision)
        return prediction_cod, prediction_si, decision, estimated_flow, y
    
if __name__ == '__main__':
    model = CIRCOD(torch.device('cuda')).cuda()
    input_tensor = torch.randn(1, 3, 512, 512).cuda()

    a,b,c,d,e,f,g = model(input_tensor, input_tensor)
