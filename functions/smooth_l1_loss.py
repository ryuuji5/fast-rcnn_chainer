import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SmoothL1Loss(function.Function):

    """Compute Smooth L1 Loss"""

    def forward_gpu(self, inputs):
        label, rect = inputs
        cuda.cupy.ElementwiseKernel(
            'raw float32 label, raw float32 rect',
            'float32 loss',
            '''
            loss = 0
            for (int i=0; i<4; i++)
            {
                float gap = rect[i] - label[i]
                if(gap > -1 || gap < 1)
                {
                    loss += 0.5 * gap * gap
                }
                else
                {
                    if(gap < 0)
                    {
                        gap = -gap
                    }
                    loss += gap - 0.5
                }
            }
            ''',
            'smooth_l1_loss'
            )(label, rect)
        return loss,

    #def backward_gpu(self, inputs, gy):

       # return bottom_diff, None

def smooth_l1_loss(label, rect):
    return SmoothL1Loss(label, rect)
