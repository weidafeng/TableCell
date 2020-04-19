# -*- coding:utf-8 -*-
# mmdet/ops/table_projection.py

# Table Projection Module
# https://github.com/weidafeng/TableCell
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init


class TableProjection(nn.Module):
    def __init__(self, in_channel, smooth=True, method='concat'):
        '''
        row and col projection. 
        :param in_channel: input
        :param smooth:  bool, if true, using 1*1 conv before projection
        :param method: concat or sum
        '''
        super(TableProjection, self).__init__()
        self.method = method
        self.smooth = smooth
        if self.smooth:
            self.smooth_conv_branch_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, dilation=(1, 2))
            self.smooth_conv_branch_2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, dilation=(2, 1))

        if self.method == 'concat':
            self.reduce_dim = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.LeakyReLU(inplace=True)

        # self.init_weights()

    def forward(self, x):
        row_p = x
        if self.smooth:
            row_p = self.smooth_conv_branch_1(row_p)
        row_p = self.row_projection(row_p)

        col_p = x
        if self.smooth:
            col_p = self.smooth_conv_branch_2(col_p)
        col_p = self.col_projection(col_p)

        # concat
        if self.method == 'concat':
            res = torch.cat([row_p, col_p], dim=1)
            # reduce dim and fusion
            res = self.reduce_dim(res)
        elif self.method == 'sum':
            res = (row_p + col_p) / 2.
        # residual path
        return (self.relu(self.bn(res)) + x)

    def col_projection(self, feature_map, down_sample=False):
        '''
        Calculate the mean value of each column as the value of all pixels in that column
        :param feature_map:  input the feature map, [N,C,H,W]
        :param down_sample:  whether to downsample
        :return: [N, C, H//2, W//2] if `down_sample=True` else [N,C,H,W]
        '''
        mean = torch.mean(feature_map, dim=2, keepdim=True)  # Calculate the average of each column

        if down_sample:
            raw_size = feature_map.shape
            return mean.expand(raw_size[0], raw_size[1], raw_size[2] // 2, raw_size[3] // 2)
        return mean.expand_as(feature_map)

    def row_projection(self, feature_map, down_sample=False):
        '''
        Calculate the mean value of each row as the value of all pixels in that row
        :param feature_map:  input the feature map, [N,C,H,W]
        :param down_sample:  whether to downsample
        :return: [N, C, H//2, W//2] if `down_sample=True` else [N,C,H,W]
        '''
        mean = torch.mean(feature_map, dim=3, keepdim=True)  # Calculate the average of each row

        if down_sample:
            raw_size = feature_map.shape
            return mean.expand(raw_size[0], raw_size[1], raw_size[2] // 2, raw_size[3] // 2)
        return mean.expand_as(feature_map)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)
