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

import os
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'fast-rcnn/lib/utils')
import time
import argparse
import cv2 as cv
import numpy as np
import cPickle as pickle
from chainer import cuda, optimizers, Variable
import chainer
print('chainer Ver = %s'% chainer.__version__)
from cython_nms import nms
from progressbar import *
from VGG import VGG
import random
from multiprocessing import Process, Queue
import logging

PIXEL_MEANS = np.array([97.566359191178279, 106.43752413089281, 92.941614945334464], dtype=np.float32)
random.seed(1741)

NETS = {'vgg16': ('VGG16',
                  'VGG16_person.chainermodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'VGG_CNN_M_1024.chainermodel'),
        'caffenet': ('CaffeNet',
                     'CaffeNet.chainermodel')}

OMPOSE_PATH = '/home/aolab/Codes/ompose'
data_dir = '%s/data/normal/images' % OMPOSE_PATH

def load_dataset(args):
    data_path = '%s/data/normal'%OMPOSE_PATH
    train_fn = '%s/det_label_train.csv' % data_path
    test_fn = '%s/det_label_test.csv' % data_path
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
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
    print('model setting')
    model = pickle.load(open('models/%s'%NETS[args.net][1]))
    model.to_gpu()

    # prepare optimizer
    #optimizer = optimizers.AdaGrad(lr=0.0005)
    optimizer = optimizers.AdaGrad(lr=0.001)
    optimizer.setup(model)

    return model, optimizer

def fliplr(args, img, train_dl, rects):
    if np.random.randint(2) == 1 and args.flip == True:
        img = cv.flip(img,1)
        img_col = np.float32(img.shape[1])
        train_dl[0] = img_col - train_dl[0]
        train_dl[2] = img_col - train_dl[2]
        rects[0] = img_col - rects[0]
        rects[2] = img_col - rects[2]
    return img, train_dl, rects

def load_data(args, input_q, data_q):
    c = args.channel
    w = 500
    h = 500

    while True:
        all_data = input_q.get()
        if all_data is None:
            break
        pbar = ProgressBar(maxval = len(all_data)).start()
        input_data = np.zeros((args.batchsize, c, h, w))
        label_data = np.zeros((args.batchsize, 4))
        obj_prs = np.zeros((args.batchsize, 5))
        cls = np.zeros((args.batchsize))
        p_count = 0
        n_count = 0
        for i, x in  enumerate(all_data):
            x=np.asarray(x.split(','))
            sample_fn = '%s/data/normal/EB_samples/%s'%(OMPOSE_PATH, x[0].replace('png','csv'))
            if os.path.exists(sample_fn):
                orig_rects = open(sample_fn).readlines()
            else:
                continue
            orig_img = cv.imread('%s/%s'%(data_dir,x[0]))
            #print(i, x[0])
            img = orig_img[50:800, 650:1400].astype(np.float32, copy=True)
            img -= PIXEL_MEANS
            #img = cv.resize(img,(500,500))
            img = cv.resize(img,(w,h), interpolation=cv.INTER_LINEAR)
            gt = np.asarray(x[1:5],dtype = np.float32)
            gt[0] -= 650 
            gt[1] -=  50
            gt[2] -= 650 
            gt[3] -=  50
            gt *= w
            gt /= 750
            c_t_x = np.average([gt[0],gt[2]])
            c_t_y = np.average([gt[1],gt[3]])
            perm = np.random.permutation(len(orig_rects))
            orig_rects = np.array(orig_rects)[perm]
            perm2 = np.random.permutation(args.batchsize)
            for j, rects in enumerate(orig_rects):
                label = np.zeros(4)
                rects = np.asarray(rects.split(','))
                IoU = float(rects[0])
                rects = np.asarray(rects[1:5], dtype = np.float32)
                rects *= w
                rects /= 750
                img, gt, rects = fliplr(args,img, gt, rects)
                c_r_x = np.average([rects[0],rects[2]])
                c_r_y = np.average([rects[1],rects[3]])
                label[0] = (c_t_x - c_r_x) / abs(rects[0]-rects[2])
                label[1] = (c_t_y - c_r_y) / abs(rects[1]-rects[3])
                label[2] = np.log(abs(rects[0]-rects[2]) / abs(gt[0]-gt[2]))
                label[3] = np.log(abs(rects[1]-rects[3]) / abs(gt[1]-gt[3]))
                if IoU > 0.5:
                    '''
                    cv.rectangle(img, (gt[0],gt[1]),(gt[2],gt[3]),(255,0,0), 2)
                    cv.rectangle(img, (rects[0],rects[1]),(rects[2],rects[3]),(0,255,0), 2)
                    cv.imshow('img',img)
                    cv.waitKey()
                    '''
                    cls[p_count + n_count -1] = 1
                    p_count += 1
                elif IoU >= 0.1 and n_count < p_count*3:
                    cls[p_count + n_count -1] = 0
                    n_count += 1
                else:
                    'none...'
                    continue
                input_data[p_count + n_count -1] = img.transpose((2,0,1))
                label_data[p_count + n_count -1] = label
                obj_prs[p_count + n_count -1] = np.hstack([0,rects])
                if p_count + n_count == args.batchsize:
                    p_count = 0
                    n_count = 0        
                    input_data = np.asarray(input_data[perm2], dtype = np.float32)
                    label_data = np.asarray(label_data[perm2], dtype = np.float32)
                    obj_prs = np.asarray(obj_prs[perm2], dtype = np.float32)
                    cls = np.asarray(cls[perm2], dtype = np.int32)
                    data_q.put([input_data, cls, obj_prs, label_data])
                    input_data = np.zeros((args.batchsize, c, h, w))
                    label_data = np.zeros((args.batchsize, 4))
                    obj_prs = np.zeros((args.batchsize, 5))
                    cls = np.zeros((args.batchsize))
                    perm2 = np.random.permutation(args.batchsize)
            pbar.update(i if (i) < len(all_data) else len(all_data))

    data_q.put([None,None,None,None])

def eval(test_dl, N, model, trans, args, input_q, data_q):
    if args.gpu >= 0:
        xp = cuda.cupy if cuda.available else np
    else:
        xp = np
    widgets = ["Eval : ", Percentage(), Bar()]
    pbar = ProgressBar(maxval = N, widgets = widgets).start()
    sum_loss = 0

    # putting all data
    input_q.put(test_dl)

    # training
    for i in xrange(0, N, args.batchsize):
        input_data, cls, obj_prs, label = data_q.get(True, None)

        input_data = xp.asarray(input_data)
        cls = xp.asarray(cls)
        rects = xp.asarray(rects)
        label = xp.asarray(label)
        cls_score,bbox_pred, l1_loss, cls_loss = model.forward(input_data, rects, cls, label, train=False)
        loss = l1_loss + cls_loss
        sum_loss += np.sum(loss.data)

    return sum_loss

def get_log_msg(stage, epoch, sum_loss, N, args, st):
    msg = 'epoch:{:02d}\t{} mean loss={}\telapsed time={} sec'.format(
        epoch + args.epoch_offset,
        stage,
        sum_loss,
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
    parser.add_argument('--batchsize','-b', type=int, default=128)
    parser.add_argument('--restart_from', '-r', type=str)
    parser.add_argument('--model', '-m', type=str, default='VGG_person')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--flip', type=bool, default=True)

    args = parser.parse_args()

    return args

def train(train_dl, N, model, optimizer, args, input_q, data_q):
    if args.gpu >= 0:
        xp = cuda.cupy if cuda.available else np
    else:
        xp = np
    widgets = ["Training : ", Percentage(), Bar()]
    sum_loss = 0
    perm = np.random.permutation(N)

    # putting all data
    print('input')
    input_q.put(train_dl[perm])
    #input_q.put(train_dl[:16])
    input_q.put(None)
    t_count=0
    # training
    while True:
        input_data, cls, rects, label = data_q.get()
        if cls == None:
            break
        input_data = xp.asarray(input_data)
        cls = xp.asarray(cls)
        rects = xp.asarray(rects)
        label = xp.asarray(label)

        optimizer.zero_grads()
        cls_score,bbox_pred, loss = model.forward(input_data[:, :, :], rects, cls, label, train=True)
        loss.backward()
        optimizer.update()
        sum_loss += np.sum(loss.data) / (args.batchsize)
        t_count += 1
    sum_loss /= t_count
    return sum_loss

if __name__ == '__main__':
    args = parse_args()

    if args.gpu >= 0:
        xp = cuda.cupy if cuda.available else np
    else:
        xp = np

    train_dl, test_dl = load_dataset(args)
    N = len(train_dl)
    N_test = len(test_dl)

    log_fn, result_dir=create_result_dir(args)
    model, optimizer = get_model_optimizer(result_dir, args)

    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    #learning loop

    n_epoch = args.epoch
    batchsize = args.batchsize

    for epoch in range(1, n_epoch + 1):
        #start data loading thread
        input_q = Queue()
        data_q = Queue()
        data_loader = Process(target=load_data, args=(args,input_q,data_q))
        data_loader.start()

        #train
        st = time.time()
        print('Training Step:', epoch)
        sum_loss= train(train_dl, N, model, optimizer, args, input_q, data_q)
        print('next training')
        msg = get_log_msg('training', epoch, sum_loss, N, args, st)
        logging.info(msg)
        print(msg)

        #quit data loading thread
        input_q.put(None)
        data_loader.join()
        print(result_dir)
        if epoch%5 == 0:
            model_fn = '%s/vgg_person_epoch_%d.chainermodel' % (
                result_dir, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)

    input_q.put(None)
    data_loader.join()

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
