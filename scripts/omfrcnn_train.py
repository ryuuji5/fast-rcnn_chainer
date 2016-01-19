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

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import dlib
from progressbar import *
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

def demo(net, image_name, classes, jnt):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(image_name)
    im = np.hstack([im,im[:,:400,:]])
    # Find object candidates
    start_time = time.time()
    rects = []
    dlib.find_candidate_object_locations(im, rects, min_size=500)
    obj_proposals = np.zeros((len(rects), 4), dtype=np.uint16)
    for i in range(len(rects)):
        obj_proposals[i][0], obj_proposals[i][1] = rects[i].left(),  rects[i].top()
        obj_proposals[i][2], obj_proposals[i][3] = rects[i].right(), rects[i].bottom()
    
    # Detect all object classes and regress object bounds

    scores, boxes = im_detect(net, im, obj_proposals)
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(time.time() - start_time, boxes.shape[0])

    jnt = np.asarray([int(p) for p in jnt])
    jnt = jnt.reshape(14,2)
    """
    if any(j > 1500 for j in jnt.T[0]) and any(j < 300 for j in jnt.T[0]):
        for i,jnt_x in enumerate(jnt.T[0]):
            if jnt_x < 300:
                jnt.T[0][i] += 2309
    x, y, w, h = cv2.boundingRect(np.asarray([jnt.tolist()]))
    cv2.rectangle(im, (x,y),(x+w,y+h),(0,255,0), 2, 4)
    #print >> label_write, os.path.basename(image_name), x, y, x+w, y+h
    if x + w < 400:
        x += 2309
        cv2.rectangle(im, (x,y),(x+w,y+h),(0,255,0), 2, 4)
        #print >> label_write, os.path.basename(image_name), x, y, x+w, y+h
    """
    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        v_img = 0
        print (dets)
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
        if dets[:, -1].any() > CONF_THRESH:
            for i,det in enumerate(dets):
                    print >> box_write, ('1, %s,%s,%d,%d,%d,%d,%f')%(cls, os.path.basename(image_name), det[0], det[1], det[2], det[3], det[4])
            #v_img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
            #plt.savefig('output/panorama/%s' % os.path.basename(image_name))
            ##plt.savefig('output/ompose/%s' % os.path.basename(image_name).replace('bmp', 'png'))

        else:
            print >> box_write, ('0, %s,%s')%(cls, os.path.basename(image_name))
            #v_img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
            print('Not detected...')
            continue

    return v_img

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    OMPOSE_PATH = '/home/aolab/Codes/ompose'
    data_dir = '%s/data/panorama/images' % OMPOSE_PATH

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))
    """
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    """
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    #print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'start...'
    #train_fn = '%s/data/normal/joint_train.csv' % OMPOSE_PATH
    #test_fn = '%s/data/normal/joint_test.csv' % OMPOSE_PATH
    test_fn = '%s/data/panorama/joint.csv' % OMPOSE_PATH
    #train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
    box_write = open('output/panorama_det_all_05.csv', 'w')
    #label_write = open('output/panorama_label.csv', 'w')
    """
    pbar = ProgressBar(maxval = len(train_dl)).start()
    print('Detect in Train Data')
    for i, datum in enumerate(train_dl):
        plt.figure()
        plt.hold(False)
        datum = datum.split(',')
        if datum[2] <= 100:
            continue
        #test_img = cv2.imread('%s/%s' % (data_dir, datum[1].replace('png', 'bmp')))
        #cv2.imshow('test_img_%d' % i, test_img)
        #cv2.waitKey()
        demo(net, '%s/%s' % (data_dir, datum[1]), ('person',), datum[3:31])
        #demo(net, '%s/%s' % (data_dir, datum[1].replace('png', 'bmp')), ('person',))
        pbar.update(pbar.currval + 1)
        #plt.show()
    """
    pbar = ProgressBar(maxval = len(test_dl)).start()
    print('Detect in Test Data')
    count = 0
    v_writer = cv2.VideoWriter('output/p_fast05_Otsuka.avi',cv2.cv.CV_FOURCC('M','J','P','G'),2,(2709,735))
    for i, datum in enumerate(test_dl):
        plt.figure()
        plt.hold(False)
        datum = datum.split(',')
        if datum[2] > 150:
            #if not datum[1] =='Go_20141211141542_Cam1_No_00091.png':
            #    continue
            #test_img = cv2.imread('%s/%s' % (data_dir, datum[1].replace('png', 'bmp')))
            #cv2.imshow('test_img_%d' % i, test_img)
            #cv2.waitKey()
            v_img = demo(net, '%s/%s' % (data_dir, datum[1]), ('person',), datum[3:31])
            #cv2.imshow('img', v_img)
            #cv2.waitKey(10)
            #v_writer.write(v_img)
            #demo(net, '%s/%s' % (data_dir, datum[1].replace('png', 'bmp')), ('person',))
            count += 1
        pbar.update(pbar.currval + 1)
        #plt.show()
    v_writer.release
    pbar.finish()
    #box_write.close()
    #label_write.close()
