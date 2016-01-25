#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'functions')
from chainer import Variable, FunctionSet
import chainer.functions as F
from roi_pooling_2d_vggM import roi_pooling_2d


class VGG_CNN_M_1024(FunctionSet):

    def __init__(self):
        super(VGG_CNN_M_1024, self).__init__(
            conv1=F.Convolution2D(3, 96, 7, stride=2, pad=0),
            conv2=F.Convolution2D(96, 256, 5, stride=2, pad=1),
            conv3=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=F.Linear(4608, 4096),
            fc7=F.Linear(4096, 1024),
            cls_score=F.Linear(4096, 21),
            bbox_pred=F.Linear(4096, 84)
        )

    def forward(self, x_data, rois, train=True):
        x = Variable(x_data, volatile=not train)
        rois = Variable(rois, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv3(h))

        h = F.relu(self.conv4(h))

        h = F.relu(self.conv5(h))
        h = roi_pooling_2d(h, rois)

        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        cls_score = F.softmax(self.cls_score(h))
        bbox_pred = self.bbox_pred(h)

        return cls_score, bbox_pred
