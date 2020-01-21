import torch
from torch import nn
from torch.nn import functional as F

from model.utils import get_norm


class Conv2dNormRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 bias=True, norm_type='Unknown'):
        super(Conv2dNormRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            get_norm(norm_type, out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class CAModule(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    code reference:
    https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    """

    def __init__(self, num_channels, *args, reduc_ratio=2, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio,
                             bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels,
                             bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # attention branch--squeeze operation
        attention = feat_map.view(feat_map.shape[0], self.num_channels,
                                -1).mean(dim=2)

        # attention branch--excitation operation
        attention = self.relu(self.fc1(attention))
        attention = self.sigmoid(self.fc2(attention))

        # attention operation
        attention = attention.view(*attention.shape, 1, 1)
        feat_map = torch.mul(feat_map, attention)

        return feat_map


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels, *args, **kwargs):
        super().__init__()
        C = num_channels
        C8 = num_channels // 8
        self.conv1 = nn.Conv2d(C, C8, kernel_size=1)
        self.conv2 = nn.Conv2d(C, C8, kernel_size=1)
        self.conv3 = nn.Conv2d(C, C, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        # C8: Reduced number of channels
        C8 = C // 8
        # A: Area
        A = H * W

        mat1 = self.conv1(feat_map).view(B, C8, A)
        mat2 = self.conv2(feat_map).view(B, C8, A)

        attention = torch.bmm(mat1.permute(0, 2, 1), mat2)
        # assert attention.shape == (B, A, A)
        attention = F.softmax(attention, dims=-1)
        mat3 = self.conv3(feat_map).view(B, C, A)
        refined_map = torch.bmm(mat3, attention.permute(0, 2, 1))
        # assert refined_map.shape == (B, C, A)
        refined_map = refined_map.view(B, C, H, W)
        feat_map = self.gamma * refined_map + feat_map
        return feat_map


class FPAModule(nn.Module):
    """
    Re-implementation of feature pyramid attention (FPA) described in:
    *Li et al., Pyramid Attention Network for Semantic segmentation, Face++2018
    """

    def __init__(self, num_channels, norm_type):
        super().__init__()

        # global pooling branch
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dNormRelu(num_channels, num_channels, kernel_size=1,
                           norm_type=norm_type)
        )

        # middle branch
        self.mid_branch = Conv2dNormRelu(num_channels, num_channels,
                                         kernel_size=1, norm_type=norm_type)

        self.downsample1 = Conv2dNormRelu(num_channels, 1, kernel_size=7,
                                          stride=2, padding=3,
                                          norm_type=norm_type)

        self.downsample2 = Conv2dNormRelu(1, 1, kernel_size=5, stride=2,
                                          padding=2, norm_type=norm_type)

        self.downsample3 = Conv2dNormRelu(1, 1, kernel_size=3, stride=2,
                                          padding=1, norm_type=norm_type)

        self.scale1 = Conv2dNormRelu(1, 1, kernel_size=7, padding=3,
                                     norm_type=norm_type)
        self.scale2 = Conv2dNormRelu(1, 1, kernel_size=5, padding=2,
                                     norm_type=norm_type)
        self.scale3 = Conv2dNormRelu(1, 1, kernel_size=3, padding=1,
                                     norm_type=norm_type)

    def forward(self, feat_map):
        height, width = feat_map.size(2), feat_map.size(3)
        gap_branch = self.gap_branch(feat_map)
        gap_branch = nn.Upsample(size=(height, width), mode='bilinear',
                                 align_corners=False)(gap_branch)

        mid_branch = self.mid_branch(feat_map)

        scale1 = self.downsample1(feat_map)
        scale2 = self.downsample2(scale1)
        scale3 = self.downsample3(scale2)

        scale3 = self.scale3(scale3)
        scale3 = nn.Upsample(size=(height // 4, width // 4), mode='bilinear',
                             align_corners=False)(scale3)
        scale2 = self.scale2(scale2) + scale3
        scale2 = nn.Upsample(size=(height // 2, width // 2), mode='bilinear',
                             align_corners=False)(scale2)
        scale1 = self.scale1(scale1) + scale2
        scale1 = nn.Upsample(size=(height, width), mode='bilinear',
                             align_corners=False)(scale1)

        feat_map = torch.mul(scale1, mid_branch) + gap_branch

        return feat_map


attention_modules = {
    'CAM': CAModule,
    'SAM': SAModule,
    'FPA': FPAModule,
    'none': lambda x, y: None
}

class AttentionMap(nn.Module):

    def __init__(self, cfg, num_channels):
        super().__init__()
        assert cfg.attention_map in attention_modules, f"""
        Unknown type '{self.cfg.attention_map}'"""
        self.map = attention_modules[cfg.attention_map](
            num_channels, cfg.norm_type)

    def forward(self, feat_map):
        return self.map(feat_map) if self.map else feat_map

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
