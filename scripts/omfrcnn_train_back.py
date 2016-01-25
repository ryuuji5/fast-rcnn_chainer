#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'fast-rcnn/lib/utils')
import time
import dlib
import argparse
import cv2 as cv
import numpy as np
import cPickle as pickle
from chainer import cuda, optimizers, Variable
from cython_nms import nms
from progressbar import *
from VGG_person import VGG
import random
from multiprocessing import Process, Queue
import logging

PIXEL_MEANS = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
random.seed(1741)

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG.chainermodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'VGG_CNN_M_1024.chainermodel'),
        'caffenet': ('CaffeNet',
                     'CaffeNet.chainermodel')}

OMPOSE_PATH = '/home/aolab/Codes/ompose'

def load_dataset(args):
    datadir = '%s/data/panorama'%OMPOSE_PATH
    train_fn = '%s/det_label_train.csv' % datadir
    test_fn = '%s/det_label_test.csv' % datadir
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
    print(len(train_dl), len(test_dl))
    return train_dl, test_dl

def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/frcnn_' + args.model
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print ('makedirs %s'%result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = 'args.result_dir'
        log_fn = '%s/log.txt'%result_dir
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir

def get_model_optimizer(result_dir, args):
    model = pickle.load(open('models/%s'%NETS[args.net][1]))
    model.to_gpu()

    # prepare optimizer
    optimizer = optimizers.AdaGrad(lr=0.0005)
    optimizer.setup(model)

    return model, optimizer

def fliplr(args, img,train_dl):
    if np.random.randint(2) == 1 and args.flip == True:
        img = np.fliplr(img)
        img_col = img.shape[2]
        train_dl[1] = img_col - train_dl[1]
        train_dl[3] = img_col - train_dl[3]
    return img, train_dl[1:5]

def eval_rects(rects, gt):
    # calc t_size
    t_heith = gt[3] - gt[1]
    t_width = gt[2] - gt[0]
    t_S = t_heith * t_width
    # calc result size
    r_heith = r_rect_bottom - r_rect_top
    r_width = r_rect_right - r_rect_left
    r_S = r_heith * r_width
    # calc IoU
    top = max(gt[1],r_rect_top)
    left = max(gt[0],r_rect_left)
    bottom = min(gt[3],r_rect_bottom)
    right = min(gt[2],r_rect_right)
    heith = bottom - top
    width = right - left
    if heith <= 0 or width <= 0:
        return -1
    else:
        S = heith * width
        IoU = float(S) / float(t_S + r_S - S)
        return IoU

def load_data(args, input_q, data_q, perm):
    c = args.channel
    w = 1000
    h = 271

    while True:
        x_batch = input_q.get()
        if x_batch is None:
            break
        input_data = np.zeros((2, c, h, w))
        label_data = np.zeros((2, 4))
        rects = np.zeros((2, 4))
        for i, x in  enumerate(x_batch):
            orig_img = cv.imread(train_dl[perm[i]])
            orig_img = np.hstack([orig_img, orig_img[:,:400,:]])
            img, gt = fliplr(args,orig_img,train_dl[perm[i]])
            img, im_scale = img_preprocessing(img, PIXEL_MEANS)
            orig_rects = get_bboxes(img, im_scale, min_size=args.min_size)
            rects[i] = eval_rects(orig_rects, gt)
            input_data[i] = img
            label_data[i] = gt
        #

        data_q.put([input_data, gt])

def eval(test_dl, N, model, trans, args, input_q, data_q):
    widgets = ["Eval : ", Percentage(), Bar()]
    pbar = ProgressBar(maxval = N, widgets = widgets).start()
    sum_loss = 0

    # putting all data
    for i in xrange(0, N, args.batchsize):
        x_batch = test_dl[i:i + args.batchsize]
        input_q.put(x_batch)

    # training
    for i in xrange(0, N, args.batchsize):
        input_data, label = data_q.get(True, None)

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        loss, pred = model.forward(input_data, label, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    return sum_loss

def get_log_msg(stage, epoch, sum_loss, N, args, st):
    msg = 'epoch:{:02d}\t{} mean loss={}\telapsed time={} sec'.format(
        epoch + args.epoch_offset,
        stage,
        sum_loss / N,
        time.time() - st)

    return msg

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--result', dest='result_dir', help='path to result_dir')
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--epoch_offset', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--restart_from', '-r', type=str)
    parser.add_argument('--model', '-m', type=str, default='VGG')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--flip', type=bool, default=True)

    args = parser.parse_args()

    return args

def train(train_dl, N, model, optimizer, args, input_q, data_q):
    widgets = ["Training : ", Percentage(), Bar()]
    pbar = ProgressBar(maxval = N, widgets = widgets).start()
    sum_loss = 0

    # putting all data
    for i in range(0, N, 2):
        x_batch = train_dl[perm[i:i + 2]]

        input_q.put(x_batch)

    # training
    for i in range(0, N, args.batchsize):
        input_data, label = data_q.get()
        print (input_data.shape, label.shape)
        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        optimizer.zero_grads()
        loss, pred = model.forward(input_data, label, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    return sum_loss


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    #img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale

def get_bboxes(orig_img, im_scale, min_size, dedup_boxes=1./16):
    rects=[]
    dlib.find_candidate_object_locations(orig_img, rects, min_size=min_size)
    rects = [[0, d.left(), d.top(), d.right(), d.bottom()] for d in rects]
    rects = np.asarray(rects, dtype=np.float32)

    # bbox pre-processing
    rects *= im_scale
    print (rects)
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(rects * dedup_boxes).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = rects[index, :]

    return rects

if __name__ == '__main__':
    args = parse_args()
    data_dir = '%s/data/panorama/images' % OMPOSE_PATH

    if args.gpu >= 0:
        xp = cuda.cupy if cuda.available else np
    else:
        xp = np

    train_dl, test_dl = load_dataset(args)
    N = len(train_dl)
    N_test = len(test_dl)

    result_dir=create_result_dir(args)
    model, optimizer = get_model_optimizer(result_dir, args)

    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    #learning loop

    n_epoch = args.epoch
    batchsize = args.batchsize
    perm = np.random.permutation(N)

    for epoch in range(1, n_epoch + 1):
        #start data loading thread
        input_q = Queue()
        data_q = Queue()
        data_loader = Process(target=load_data, args=(args,input_q,data_q, perm))
        data_loader.start()

        #train
        sum_loss= train(train_dl, N, model, optimizer, args, input_q, data_q)
        msg = get_log_msg('training', epoch, sum_loss, N, args, st)
        logging.info(msg)
        print(msg)

        #quit data loading thread
        input_q.put(None)
        data_loader.join()

    model_fn = '%s/%s_epoch_%d.chainermodel' % (
        result_dir, args.prefix, epoch + args.epoch_offset)
    pickle.dump(model, open(model_fn, 'wb'), -1)

    input_q.put(None)
    data_loader.join()

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
