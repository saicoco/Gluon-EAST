# coding=utf-8
import gluoncv
from model_zoo import east
from mxnet import nd as F
import mxnet as mx
import cv2
import numpy as np

if __name__ == '__main__':

    x = F.random_normal(shape=(1, 3, 512, 512))
    model = east.EAST(nclass=2)

    model.load_parameters(filename='/Users/gengjiajia/.mxnet/models/resnet50_v1b-e263a986.params', allow_missing=True, ignore_extra=True)
    model.collect_params("decoder*|east_head").initialize(init=mx.init.Xavier(), verbose=True)
    fscore, fgeo = model.forward(x)
    print F.min(fscore), F.min(fgeo)
    print("fscore:{}, fgeo:{}".format(fscore.shape, fgeo.shape))
    cv2.imwrite('score_map.png', fscore[0,0].asnumpy()*255)