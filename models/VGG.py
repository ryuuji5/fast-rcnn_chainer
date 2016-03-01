#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'functions')
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F
import numpy as np
from roi_pooling_2d import roi_pooling_2d
from smooth_l1_loss import smooth_l1_loss


class VGG(FunctionSet):

    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=F.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=F.Linear(25088, 4096),
            fc7=F.Linear(4096, 4096),
            cls_score=F.Linear(4096, 2),
            bbox_pred=F.Linear(4096, 4)
        )

    def forward(self, x_data, r, c, gt, train=True):
        x = Variable(x_data, volatile=not train)
        rois = Variable(r, volatile=not train)
        t = Variable(gt, volatile=not train)
        cls = Variable(c, volatile=not train)
        cls_mask = Variable(cuda.cupy.asarray(c,dtype='f'), volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = roi_pooling_2d(h, rois)
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        cls_prob = F.softmax(self.cls_score(h))
        bbox_pred = self.bbox_pred(h)
        #cls_loss = F.softmax_cross_entropy(self.cls_score(h), cls)
        cls_loss = -F.log(cls_prob)
        l1_loss = smooth_l1_loss(bbox_pred, t)
        cls_loss = F.select_item(cls_loss, cls)
        loss = cls_loss + l1_loss*cls_mask

        return cls_prob, bbox_pred, loss