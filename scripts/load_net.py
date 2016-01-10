#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../fast-rcnn/tools')
sys.path.insert(0, 'models')
sys.path.insert(0, '../fast-rcnn/caffe-fast-rcnn/build/install/python')
import _init_paths
import caffe
from caffenet import CaffeNet
from VGG import VGG
from VGG_CNN_M_1024 import VGG_CNN_M_1024
import cPickle as pickle
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import os

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}
net = NETS['vgg_cnn_m_1024']
model_name = net[0]
print (net)
param_dir = 'fast-rcnn/data/fast_rcnn_models'
param_fn = '%s/%s' % (param_dir, net[1])
model_dir = 'fast-rcnn/models/%s' % net[0]
model_fn = '%s/test.prototxt' % model_dir

prototxt = os.path.join(cfg.ROOT_DIR, 'models', net[0], 'test.prototxt')
caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',net[1])

caffenet = CaffeNet()
net = caffe.Net(prototxt, caffemodel, caffe.TEST)
for name, param in net.params.iteritems():
    layer = getattr(caffenet, name)

    print name, param[0].data.shape, param[1].data.shape,
    print layer.W.shape, layer.b.shape

    layer.W = param[0].data
    layer.b = param[1].data
    setattr(caffenet, name, layer)

pickle.dump(caffenet, open('models/%s.chainermodel'%model_name, 'wb'), -1)
