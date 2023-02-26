# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from Attention.siamAM import simam_module
from mmdet.models.builder import NECKS

@NECKS.register_module()
class mobileneck(BaseModule):
    def __init__(self,
                 num_deconv_kernels =[4, 4, 4],
                 use_dcn=True,
                 init_cfg=None):
        super(mobileneck, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.num_deconv_kernels = num_deconv_kernels
        self.deconv1=self._make_deconv_layer(512, 256, self.num_deconv_kernels[0])
        self.deconv2 = self._make_deconv_layer(256, 128, self.num_deconv_kernels[1])
        self.deconv3 = self._make_deconv_layer(128, 64, self.num_deconv_kernels[2])
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
        outs = []
        C0 = inputs[-4]
        C1 = inputs[-3]
        C2 = inputs[-2]
        C3 = inputs[-1]
        '''特征融合'''
        P3 = self.deconv1(C3) + C2
        P2 = self.deconv2(P3) + C1
        P1 = self.deconv3(P2) + C0
        P1 = self.relu(P1)

        return P1,
