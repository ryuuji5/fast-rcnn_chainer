#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../fast-rcnn/tools')
sys.path.insert(0, 'models')
sys.path.insert(0, '../fast-rcnn/caffe-fast-rcnn/build/install/python')
import _init_paths
import caffe
from CaffeNet_person import CaffeNet
#from VGG_CNN_M_1024 import VGG_CNN_M_1024
import cPickle as pickle
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import os

def copy_model(src, dst):
  assert isinstance(src, link.Chain)
  assert isinstance(dst, link.Chain)
  for child in src.children():
      if child.name not in dst.__dict__: continue
      dst_child = dst[child.name]
      if type(child) != type(dst_child): continue
      if isinstance(child, link.Chain):
          copy_model(child, dst_child)
      if isinstance(child, link.Link):
          match = True
          for a, b in zip(child.namedparams(), dst_child.namedparams()):
              if a[0] != b[0]:
                  match = False
                  break
              if a[1].data.shape != b[1].data.shape:
                  match = False
                  break
          if not match:
              print 'Ignore %s because of parameter mismatch' % child.name
              continue
          for a, b in zip(child.namedparams(), dst_child.namedparams()):
              b[1].data = a[1].data
          print 'Copy %s' % child.name

if __name__=="__main__":
  NETS = {'vgg16': ('VGG16',
                    'vgg16_fast_rcnn_iter_40000.caffemodel'),
          'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                             'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
          'caffenet': ('CaffeNet',
                       'caffenet_fast_rcnn_iter_40000.caffemodel')}
  net_choice = NETS['caffenet']
  param_dir = 'fast-rcnn/data/fast_rcnn_models'
  param_fn = '%s/%s' % (param_dir, net_choice[1])
  model_dir = 'fast-rcnn/models/%s' % net_choice[0]
  model_fn = '%s/test.prototxt' % model_dir

  prototxt = os.path.join(cfg.ROOT_DIR, 'models', net_choice[0], 'test.prototxt')
  caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',net_choice[1])

  #caffenet = VGG()
  caffenet = CaffeNet()
  #caffenet = vgg_cnn_m_1024()
  #caffenet = VGG_CNN_M_1024()
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  for name, param in net.params.iteritems():
      layer = getattr(caffenet, name)
      
      if param[0].data.shape == layer.W.shape and param[1].data.shape == layer.b.shape:
        print name, param[0].data.shape, param[1].data.shape,
        print layer.W.shape, layer.b.shape
        layer.W = param[0].data
        layer.b = param[1].data
      else:
        print('not copy', name, param[0].data.shape, param[1].data.shape,layer.W.shape, layer.b.shape)
      print('setting',layer.W.shape, layer.b.shape)
      setattr(caffenet, name, layer)

  #vgg_new = VGG() 
  #copy_model(caffenet, vgg_new)
  pickle.dump(caffenet, open('models/alex_person.chainermodel', 'wb'), -1)
