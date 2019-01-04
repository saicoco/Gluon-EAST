# coding=utf-8
from __future__ import absolute_import
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from gluoncv.model_zoo.segbase import SegBaseModel
from mxnet import cpu
import numpy as np
from mxnet.gluon.data.vision import transforms

from mxnet import nd as F

class EAST(SegBaseModel):
    def __init__(self, nclass, text_scale=512, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=512, **kwargs):
        super(EAST, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)

        self.head = _EAST_head(text_scale=text_scale, prefix='east_head')
        # [64, 128, 256, 512]
        self.conv_stage1 = nn.Conv2D(128, 1, prefix='decoder_conv1')
        self.conv_stage1_3 = nn.Conv2D(128, 3, padding=(1, 1), prefix='decoder_conv1_1')

        self.conv_stage2 = nn.Conv2D(64, 1, prefix='decoder_conv2')
        self.conv_stage2_3 = nn.Conv2D(64, 3, padding=(1, 1), prefix='decoder_conv2_1')

        self.conv_stage3 = nn.Conv2D(32, 1, prefix='decoder_conv3')
        self.conv_stage3_3 = nn.Conv2D(32, 3, padding=(1, 1), prefix='decoder_conv3_1')
        self.crop_size = crop_size

    def base_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return x, c2, c3, c4

    def hybrid_forward(self, F, x, *args, **kwargs):
        h, w = x.shape[2:]
        # unet
        c1, c2, c3, c4 = self.base_forward(x)
        # stage 5
        # g0 = F.contrib.BilinearResize2D(c4, self.crop_size//16, self.crop_size//16)
        g0 = c4
        c1_1 = self.conv_stage1(F.Concat(g0, c3, dim=1))
        h1 = self.conv_stage1_3(c1_1)

        g1 = F.contrib.BilinearResize2D(h1, h//8, w//8)
        c2_2 = self.conv_stage2(F.Concat(g1, c2, dim=1))
        h2 = self.conv_stage2_3(c2_2)

        g2 = F.contrib.BilinearResize2D(h2, h//4, w//4)
        c3_3 = self.conv_stage3(F.Concat(g2, c1))
        h3 = self.conv_stage3_3(c3_3)

        F_score, F_geometry = self.head(h3)

        return F_score, F_geometry

class _EAST_head(HybridBlock):

    def __init__(self, text_scale=512, prefix=None, params=None):
        super(_EAST_head, self).__init__(prefix=prefix, params=params)
        self.text_scale = text_scale
        with self.name_scope():
            self.score_branch = nn.Conv2D(1, 1, activation='sigmoid')
            self.geo_branch = nn.Conv2D(4, 1, activation='sigmoid')
            self.theta_branch = nn.Conv2D(1, 1, activation='sigmoid')


    def hybrid_forward(self, F, x, *args, **kwargs):
        score_map = self.score_branch(x)
        geo_map = self.geo_branch(x) * self.text_scale
        angle_map = self.theta_branch(x) * np.pi / 2.
        geometry_map = F.Concat(geo_map, angle_map, dim=1)

        return score_map, geometry_map

