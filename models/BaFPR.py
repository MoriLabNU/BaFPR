import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b2
#from .scSE_block import ChannelSpatialSELayer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

#! part of model taken from https://github.com/DengPingFan/Polyp-PVT/blob/57394b7bc20b171aa07884b110cd5a51919f3448/lib/pvt.py

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,groups=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CAB(nn.Module):
    def __init__(self, channel):
        super(CAB, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_downsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample5 = BasicConv2d(channel * 2, channel * 2, 3, padding=1)
        self.conv_x1_down = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_x2_down = BasicConv2d(channel, channel, 3, padding=1)


        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.conv_concat2_down = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3_down = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4_down = BasicConv2d(3 * channel, channel, 3, padding=1)

        self.conv5 =  BasicConv2d(2 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        

        x2_1_down = self.conv_downsample1(self.downsample(x3)) * self.conv_x2_down(x2)
        #print(self.conv_downsample2(self.downsample(self.downsample(x3))).shape, self.conv_downsample3(self.downsample(x2)).shape, x1.shape)
        x1_1_down =  self.conv_downsample2(self.downsample(self.downsample(x3))) * self.conv_downsample3(self.downsample(x2)) * self.conv_x1_down(x1)

        x2_2_down = self.conv_concat2_down(torch.cat((x2_1_down, self.conv_downsample4(self.downsample(x3))), 1))
        #print(x1_1_down.shape, self.conv_downsample5(self.downsample(x2_2_down)).shape)
        x1_2_down = self.conv_concat3_down(torch.cat((x1_1_down, self.conv_downsample5(self.downsample(x2_2_down))), 1))

        x1_down = self.conv4_down(x1_2_down)
        x1 = self.conv4(x3_2)

        #print(x1_down.shape, x1.shape)

        x1_fusion = self.conv5(torch.cat((self.upsample_4(x1_down), x1), 1))





        return x1_fusion









class fusion_module(nn.Module):
    def __init__(self, num_in=32, plane_mid=256, mids=4, normalize=False):
        super(fusion_module, self).__init__()

        self.num_s = int(plane_mid)
        #self.num_n = (mids) * (mids)

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=9, padding=4, groups=self.num_s//num_in)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=9, padding=4, groups=self.num_s//num_in)
        self.conv_fusion =  nn.Conv2d(self.num_s, self.num_s, kernel_size=1)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        
        self.multi_attn = nn.MultiheadAttention(num_in, 8)

    def forward(self, x1, x2 ):
        b, c, h, w = x1.shape

        x1 = self.conv_state(x1)
        x2 = self.conv_proj(x2)
        
        #x_attn, x_attn_weight = self.multi_attn(x1.reshape(b, c, -1).permute(2, 0, 1), x1.reshape(b, c, -1).permute(2, 0, 1), x2.reshape(b, c, -1).permute(2, 0, 1))
        #x_attn = x_attn.permute(1,2,0).reshape(b, self.num_s, h, w)
        x = self.conv_fusion(x1+x2)
        x = self.conv_extend(x)

        return x



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


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


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return 



class dist_head(nn.Module):
    def __init__(self, in_chn, out_chn=1):
        super(dist_head, self).__init__()
        self.branch = nn.Sequential(
            BasicConv2d(in_chn, 256, 3),
            nn.ReLU(),
            nn.Dropout(0.5),  
            BasicConv2d(256, out_chn, 3),
            nn.ReLU())

        self.dist_out = nn.Conv2d(out_chn, 1, kernel_size=1, stride=1, bias=True)

        self._init_weight()

    def forward(self, x):
        x = self.branch(x)
        x = self.dist_out(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class BaFPR(nn.Module):
    def __init__(self, channel=32, **kwargs):
        super(BaFPR, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './assets/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        #self.CFM = CFM(channel) # single direction
        self.CFM = CAB(channel) # bi-directional
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        #self.scSE = ChannelSpatialSELayer(64)
        self.fusion = fusion_module()
        self.dist = BasicConv2d(64, channel, 1)
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

        self.out_dist = dist_head(channel)



    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        

        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention



        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)


        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)



        dist = self.dist(cim_feature)
        sam_feature = self.fusion(cfm_feature, T2)


        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)
        prediction3 = self.out_dist(dist)


        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        prediction3_8 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')  
        return prediction1_8, prediction2_8, prediction3_8














if __name__ == '__main__':
    model = BaFPR().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2, prediction3 = model(input_tensor)
    print(prediction1.size(), prediction2.size())