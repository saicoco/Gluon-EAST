# coding=utf-8
from model_zoo import east
import mxnet as mx
import cv2
import glob, os, sys
import numpy as np
import time
import lanms
from data.utils import restore_rectangle
from mxnet.gluon.data.vision import transforms
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('im_dir', type=str, default='demo_images/')
parser.add_argument('ckpt_path', type=str, default='ckpt/model_32992.params')
parser.add_argument('out_dir', type=str, default='result')
parser.add_argument('gpuid', type=int, default=-1)
args = parser.parse_args()

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.6, box_thresh=0.1, nms_thres=0.3):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def main(im_dir, ckpt_path, ctx, out_dir='result', write_images=True):
    east_model = east.EAST(nclass=2, text_scale=1024)
    ctx = mx.cpu() if ctx < 0 else mx.gpu()
    # east_model.hybridize()
    east_model.load_parameters(ckpt_path, ctx)
    east_transform = transforms.Compose([
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    imlist = glob.glob1(im_dir, '*g')
    for i, im_name in enumerate(imlist):
        im_path = os.path.join(im_dir, im_name)
        start_time = time.time()
        im = cv2.imread(im_path)
        im_resized, (ratio_h, ratio_w) = resize_image(im, max_side_len=784)
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        im_resized = east_transform(mx.nd.array(im_resized.transpose((2, 0, 1))))

        f_score, f_geometry = east_model.forward(im_resized.expand_dims(axis=0))
        timer['net'] = time.time() - start

        score_map = f_score.asnumpy().transpose((0, 2, 3, 1))
        cv2.imwrite("score_map{}.png".format(i), score_map[0, :, :, 0]*255)
        geo_map = f_geometry.asnumpy().transpose((0, 2, 3, 1))
        boxes, timer = detect(score_map, geo_map, timer)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            im_name, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        # save to file
        if boxes is not None:
            res_file = os.path.join(
                out_dir,
                '{}.txt'.format(
                    os.path.basename(im_path).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(im[:, :, ::-1].astype(np.uint8), [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                                  thickness=1)
        if write_images:
            img_path = os.path.join(out_dir, im_name)
            cv2.imwrite(img_path, im)
if __name__ == '__main__':
    main(im_dir=args.im_dir, ckpt_path=args.ckpt_path, ctx=args.gpuid, out_dir=args.out_dir)
