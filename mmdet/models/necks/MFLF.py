# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from Attention.CA import CoordAtt
from mmdet.models.builder import NECKS
from Attention.SCA import SCA

'''卷积'''
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

@NECKS.register_module()
class MFLF(BaseModule):
    def __init__(self,
                 use_dcn=True,
                 init_cfg=None):
        super(MFLF, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn

        self.sca1 = SCA(512, 128)
        self.sca2 = SCA(1024, 256)
        self.sca3 = SCA(2048, 512)

        self.deconv1 = self._make_deconv_layer(512, 256, 4)
        self.deconv2 = self._make_deconv_layer(256, 128, 4)
        self.deconv3 = self._make_deconv_layer(128, 64, 4)

        self.relu=nn.ReLU()
    def _make_deconv_layer(self, inchannel, outchannel,deconv_kernel):
        '''
        创建一个deconv层,由一个3*3的可行变卷积模块和一个deconv上采样模块构成。尺寸放大一倍
        '''
        layers = []
        conv_module = ConvModule(inchannel,
                                 outchannel,
                                 kernel_size=3,
                                 padding=1,
                                 conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                                 norm_cfg=dict(type='BN')
                                 )
        layers.append((conv_module))
        upsample_module = ConvModule(
            outchannel,
            outchannel,
            deconv_kernel,
            stride=2,
            padding=1,
            conv_cfg=dict(type='deconv'),
            norm_cfg=dict(type='BN'))
        layers.append(upsample_module)
        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        C1 = inputs[-3]
        C2 = inputs[-2]
        C3 = inputs[-1]
        '''Multi-scale Lacation Fusion'''
        K1 = self.sca1(C1)

        K2 = self.sca2(C2)

        K3 = self.sca3(C3)

        P1 = self.deconv1(K3) + K2

        P2 = self.deconv2(P1) + K1

        P3= self.deconv3(P2)

        return P3,
