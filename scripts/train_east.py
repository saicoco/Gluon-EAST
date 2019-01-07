# coding=utf-8
import mxnet as mx
from mxnet import gluon, autograd
import gluoncv as gcv
from model_zoo import east, EASTLoss
from data.ic_data import text_detection_data
from mxnet.gluon.data import DataLoader
from mxnet.gluon import utils
import logging
import os, sys
from mxboard import SummaryWriter
import numpy as np

logging.basicConfig(level=logging.INFO)

def main(train_dir, ctx=None, lr=0.0001, epoches=20, batch_size=16, checkpoint_path='model', debug=False):
    summ_writer = SummaryWriter(checkpoint_path)
    # dataloader
    ctx = [mx.cpu()] if not ctx else [mx.gpu(eval(i)) for i in ctx]
    ic_data = text_detection_data(image_dir=train_dir)
    ic_dataloader = DataLoader(dataset=ic_data, batch_size=batch_size, shuffle=True, num_workers=16)
    data_num = len(ic_dataloader) * batch_size
    # model
    east_model = east.EAST(nclass=2, text_scale=1024, ctx=ctx)
    # east_model.load_parameters(filename='/Users/gengjiajia/.mxnet/models/resnet50_v1b-e263a986.params', allow_missing=True,
    #                       ignore_extra=True)
    east_model.collect_params().initialize(init=mx.init.Xavier(), verbose=True, ctx=ctx)
    if not debug:
        east_model.hybridize()
    trainer = gluon.Trainer(east_model.collect_params(),
                            'sgd',
                            {'learning_rate': lr,
                             'wd': 1e-5,
                             'momentum': 0.9,
                             'clip_gradient': 5}
                            )
    EAST_loss = EASTLoss(cls_weight=0.01, iou_weight=1.0, angle_weight=20)
    step = 0
    lr_counter = 0
    lr_steps = [5, 10, 15, 20]
    lr_factor = 0.9

    for epoch in range(epoches):
        loss = []
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1
        for i, batch_data in enumerate(ic_dataloader):
            im, score_map, geo_map, training_mask = map(lambda x: utils.split_and_load(x, ctx), batch_data)


            with autograd.record(train_mode=True):
                for im_x, score_map_x, geo_map_x, training_mask_x in zip(im, score_map, geo_map, training_mask):
                    f_score, f_geo = east_model(im_x)
                    batch_loss = EAST_loss(score_map_x, f_score, geo_map_x, f_geo, training_mask_x)
                    loss.append(batch_loss)

                for bl in loss:
                    bl.backward()

            trainer.step(batch_size)
            # if i % 2 == 0:
            step = epoch * data_num  + i * batch_size
            model_loss = np.mean(map(lambda x: x.asnumpy()[0], loss))
            summ_writer.add_scalar('model_loss', model_loss)
            logging.info("step: {}, loss: {}".format(step, batch_loss.asnumpy()))
        ckpt_file = os.path.join(checkpoint_path, "model_{}.params".format(step))
        east_model.collect_params().save_parameters(ckpt_file)
        logging.info("save model to {}".format(ckpt_file))

if __name__ == '__main__':
    train_dir = sys.argv[1]
    ckpt_path = sys.argv[2]
    ctxes = sys.argv[3]
    ctxes = [map(eval, ctxes.split(','))]
    main(train_dir=train_dir, ctx=ctxes, checkpoint_path=ckpt_path)
