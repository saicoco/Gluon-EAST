# coding=utf-8
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
from gluoncv import model_zoo
from gluoncv.model_zoo.resnetv1b import resnet50_v1b
from feature import FPNFeatureExpander

class EAST(HybridBlock):

    def __init__(self, text_scale=1024, ctx=mx.cpu(), pretrained=True,  **kwargs):
        super(EAST, self).__init__()
        self.text_scale = text_scale
        base_network = resnet50_v1b(pretrained=pretrained, dilated=False,
                                use_global_stats=True, ctx=ctx, **kwargs)
        self.features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=False, no_bias=False, pretrained=pretrained, ctx=ctx)
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2.)
        self.head = _EAST_head(text_scale=text_scale, prefix='east_head')
        self.decoder_out = nn.HybridSequential(prefix='decoder_out')
        with self.decoder_out.name_scope():
            self.decoder_out.add(nn.Conv2D(128, 3, 1, 1))
            self.decoder_out.add(nn.BatchNorm())
            self.decoder_out.add(nn.Activation('relu'))
            self.decoder_out.add(nn.Conv2D(64, 3, 1, 1))
            self.decoder_out.add(nn.BatchNorm())
            self.decoder_out.add(nn.Activation('relu'))
            self.decoder_out.initialize(weight_init, ctx=ctx)

    def hybrid_forward(self, F, x, **kwargs):
        # output: c4 -> c1 [1/4, 1/8, 1/16. 1/32]
        fpn_features = self.features(x)[0]
        score_map, geometry_map = self.head(fpn_features)
        return score_map, geometry_map

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
        angle_map = (self.theta_branch(x) - 0.5) * np.pi / 2.
        geometry_map = F.Concat(geo_map, angle_map, dim=1)

        return score_map, geometry_map


if __name__ == '__main__':
    import numpy as np
    fpn = EAST(pretrained=True)
    fpn.initialize(ctx=mx.cpu())
    x = mx.nd.array([np.random.normal(size=(3, 512, 512))])
    print map(lambda x:x.shape, fpn(x))


