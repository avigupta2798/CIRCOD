import torch
import torch.nn as nn
from models.pvtv2 import pvt_v2_b2

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

class SEN(nn.Module):
    def __init__(self, device='cpu', channel=64):
        super(SEN, self).__init__()
        self.backbone = pvt_v2_b2()
        path = './pre_trained_models/pvt_v2_b2.pth'
        in_channel = [64,128,320,512]
        self.Translayer2_1 = BasicConv2d(in_channel[1], channel, 3, padding=1)
        self.Translayer2_0 = BasicConv2d(in_channel[0], channel, 3, padding=1)
        self.Translayer3_1 = BasicConv2d(in_channel[2], channel, 3, padding=1)
        self.Translayer4_1 = BasicConv2d(in_channel[3], channel, 3, padding=1)
        save_model = torch.load(path, map_location=device)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.sigmoid = nn.Sigmoid()
        self.LFM = LFM()
        self.upsample_1 = nn.ConvTranspose2d(in_channels=128, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        pvt_x = self.backbone(x)
        x1 = pvt_x[0]
        x2 = pvt_x[1]
        x3 = pvt_x[2]
        x4 = pvt_x[3]
        x4_t = self.Translayer4_1(x4)
        x3_t = self.Translayer3_1(x3)
        x2_t = self.Translayer2_1(x2)
        x1_t = self.Translayer2_0(x1)
        LFM_feature = self.LFM(x1_t, x2_t, x3_t, x4_t)
        prediction1 = self.upsample_1(LFM_feature)
        pred = self.upsample_2(prediction1)
        prediction1_8 = self.sigmoid(pred)
        new = x*prediction1_8
        new_2 = (1-prediction1_8)
        new_3 = 1-x
        new_4 = new_2*new_3
        new_final = new+new_4
        return new_final, prediction1_8, pred
