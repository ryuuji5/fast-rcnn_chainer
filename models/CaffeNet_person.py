#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'functions')
from chainer import Variable, FunctionSet
import chainer.functions as F
from roi_pooling_2d_vggM import roi_pooling_2d
from smooth_l1_loss import smooth_l1_loss


class CaffeNet(FunctionSet):

    """
    Single-GPU AlexNet with Normalization layers replaced by BatchNormalization.
    """
    def __init__(self):
        super(CaffeNet, self).__init__(
            conv1=F.Convolution2D(3,  96, 11, stride=4, pad=5),
            bn1=F.BatchNormalization(96),
            conv2=F.Convolution2D(96, 256,  5, stride=1, pad=2),
            bn2=F.BatchNormalization(256),
            conv3=F.Convolution2D(256, 384,  3, stride=1,  pad=1),
            conv4=F.Convolution2D(384, 384,  3, stride=1,  pad=1),
            conv5=F.Convolution2D(384, 256,  3, stride=1,  pad=1),
            fc6=F.Linear(9216, 4096),
            fc7=F.Linear(4096, 4096),
            cls_score=F.Linear(4096, 2),
            bbox_pred=F.Linear(4096, 4)
        )

    def forward(self, x_data, r, c, gt, train=True):
        x = Variable(x_data, volatile=not train)
        rois = Variable(r, volatile=not train)
        t = Variable(gt, volatile=not train)
        cls = Variable(c, volatile=not train)

        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 3, stride=2, pad=1)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 3, stride=2, pad=1)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = roi_pooling_2d(h, rois)

        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        cls_prob = F.softmax(self.cls_score(h))
        bbox_pred = self.bbox_pred(h)
        #cls_loss = F.softmax_cross_entropy(self.cls_score(h), cls)
        cls_loss = -F.log(cls_prob)
        l1_loss = smooth_l1_loss(bbox_pred, t)
        cls_loss = F.select_item(cls_loss, cls)
        loss = cls_loss + l1_loss
        return cls_prob, bbox_pred, loss