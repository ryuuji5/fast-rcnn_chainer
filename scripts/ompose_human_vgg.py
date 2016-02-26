#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'fast-rcnn/lib/utils')
import time
import dlib
import argparse
import cv2 as cv
import numpy as np
import cPickle as pickle
from VGG import VGG
from chainer import cuda
from cython_nms import nms
from progressbar import *

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)


def get_model():
    vgg = pickle.load(open('results/test/vgg_person_epoch_14.chainermodel'))
    #vgg.to_gpu()
    if args.gpu >= 0:
        vgg.to_gpu()

    return vgg


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale

def rect_argument(rect):
    center_x = float(rect[0]+rect[2]) /2
    center_y = float(rect[0]+rect[2]) /2
    width = abs(rect[2] - rect[0])
    height = abs(rect[3] - rect[1])


def eval_rects(rects, gt):
    # calc t_size
    t_heith = gt[3] - gt[1]
    t_width = gt[2] - gt[0]
    t_S = t_heith * t_width
    # calc result size
    r_heith = rects[3] - rects[1]
    r_width = rects[2] - rects[0]
    r_S = r_heith * r_width
    # calc IoU
    top = max(gt[1],rects[1])
    left = max(gt[0],rects[0])
    bottom = min(gt[3],rects[3])
    right = min(gt[2], rects[2])
    heith = bottom - top
    width = right - left
    if heith <= 0 or width <= 0:
        return -1
    else:
        S = heith * width
        IoU = float(S) / float(t_S + r_S - S)
        return IoU

def draw_result(out, clss, bbox, rects, nms_thresh, conf, im_name, gt):
    conf = 0.9
    box_write = open('output/normal_boxes/%s'%im_name.replace('png','csv'), 'a')
    dets = np.hstack((bbox, clss[:,0][:,np.newaxis]))    
    keep = nms(dets, nms_thresh)
    dets = dets[keep, :]
    orig_rects = cuda.cupy.asnumpy(rects)[keep, 1:]
    inds = np.where(dets[:, -1] >= conf)[0]
    if len(inds) == 0:
        print('not detected...')
    cv.rectangle(out, (gt[0],gt[1]),(gt[2],gt[3]),(0,255,0),2)
    #cv.imshow('out',out)
    #cv.waitKey()
    for i in inds:
        _bbox = dets[i, :4]
        print(_bbox)
        #orig_rects is rects got in method of obj_prs(ex. SS, EB...)
        #this time I use rects got from EB
        x1, y1, x2, y2 = orig_rects[i]
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        center_x = x1 + 0.5 * width
        center_y = y1 + 0.5 * height
        dx, dy, dw, dh = map(int, _bbox)
        
        _center_x = dx * width + center_x
        _center_y = dy * height + center_y
        _width = np.exp(dw) * width
        _height = np.exp(dh) * height
        x1 = _center_x - 0.5 * _width
        y1 = _center_y - 0.5 * _height
        x2 = _center_x + 0.5 * _width
        y2 = _center_y + 0.5 * _height
        print(('%s,%s,%s,%s,%s') %(x1,y1,x2,y2,dets[i,4]))
        print >> box_write, ('%s,%s,%s,%s,%s') %(x1,y1,x2,y2,dets[i,4])

        cv.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 2)
    return out


if __name__ == '__main__':
    OP_PATH = '/home/aolab/Codes/ompose'
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    test_fn = '%s/data/normal/det_label_test.csv' % OP_PATH
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
    if args.gpu >= 0:
        xp = cuda.cupy if cuda.available else np
    else:
        xp = np
    pbar = ProgressBar(maxval = len(test_dl)).start()
    vgg = get_model()
    for i, datum in enumerate(test_dl):
        pbar.update(pbar.currval + 1)            
        datum = datum.split(',')
        im_name = datum[0]
        sample_fn = '%s/data/normal/EB_samples/%s'%(OP_PATH, im_name.replace('png','csv'))

        gt = np.zeros((len(EB_rects), 4), dtype = np.float32)
        for r, rect in enumerate(EB_rects):
            EB_rects[r] = rect.strip().split(',')
            EB_rects[r] = np.asarray(EB_rects[r][0:4], dtype = np.float32)
            EB_rects[r] = np.hstack([0,EB_rects[r]])
        EB_rects = np.asarray(EB_rects, dtype = np.float32)

        orig_image = cv.imread('%s/data/normal/images/%s'%(OP_PATH, im_name))
        clss = np.ones(len(EB_rects), dtype = np.int32)
        start_time = time.time()
        crop_img = orig_image[50:800, 650:1400]
        img = crop_img.transpose((2,0,1))
        img = xp.asarray(img, dtype = 'f')
        rects = xp.asarray(EB_rects)
        clss = xp.asarray(clss)
        gt = xp.asarray(gt)
        cls_score, bbox_pred, l1_loss, cls_loss= vgg.forward(img[xp.newaxis, :, :, :], rects, clss, gt)
        print('detection took {:.3f}s').format(time.time()-start_time)

        clss = cuda.cupy.asnumpy(cls_score.data) if args.gpu >= 0 else cls_score.data
        bbox = cuda.cupy.asnumpy(bbox_pred.data) if args.gpu >= 0 else bbox_pred.data
        gt = np.asarray(datum[1:5], dtype = np.float32)
        result = draw_result(crop_img, clss, bbox, rects, args.nms_thresh, args.conf, im_name, gt)
        cv.imwrite('output/normal/%s'%datum[0], result)
    pbar.finish()
