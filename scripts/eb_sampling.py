import sys
import numpy as np
from progressbar import *
import csv
import argparse
import cv2 as cv

OMPOSE_PATH = '/home/aolab/Codes/ompose'
data_dir = '%s/data/normal/images' % OMPOSE_PATH

def load_dataset(args):
    data_path = '%s/data/normal'%OMPOSE_PATH
    train_fn = '%s/det_label_train.csv' % data_path
    test_fn = '%s/det_label_test.csv' % data_path
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
    return train_dl, test_dl

def eval_rects(label, rect):
    t_rect_bottom = label[3]
    t_rect_top = label[1]
    t_rect_right = label[2]
    t_rect_left = label[0]
    r_rect_bottom = rect[3]
    r_rect_top = rect[1]
    r_rect_right = rect[2]
    r_rect_left = rect[0]
    # calc t_size
    t_heith = t_rect_bottom - t_rect_top
    t_width = t_rect_right - t_rect_left
    t_S = t_heith * t_width
    # calc result size
    r_heith = r_rect_bottom - r_rect_top
    r_width = r_rect_right - r_rect_left
    r_S = r_heith * r_width
    # calc IoU
    top = max(t_rect_top,r_rect_top)
    left = max(t_rect_left,r_rect_left)
    bottom = min(t_rect_bottom,r_rect_bottom)
    right = min(t_rect_right,r_rect_right)
    heith = bottom - top
    width = right - left
    if heith <= 0 or width <= 0:
        return -1
    else:
        S = heith * width
        IoU = float(S) / float(t_S + r_S - S)
        return IoU

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--result', dest='result_dir', help='path to result_dir')
    parser.add_argument('--min_size', type=int, default=500)
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--epoch_offset', type=int, default=0)
    parser.add_argument('--batchsize','-b', type=int, default=128)
    parser.add_argument('--restart_from', '-r', type=str)
    parser.add_argument('--model', '-m', type=str, default='Alex_person')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--flip', type=bool, default=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    train_dl, test_dl = load_dataset(args)
    pbar = ProgressBar(maxval = len(train_dl)).start()
    p_count = 0
    all_count = 0
    for i, datum in enumerate(train_dl):
        pbar.update(pbar.currval + 1)            
        datum = np.asarray(datum.strip().split(','))
        im_name = datum[0]
        gt = np.asarray(datum[1:], dtype = int)
        gt[0] -= 650
        gt[1] -= 50
        gt[2] -= 650
        gt[3] -= 50
        ebs = open('%s/data/EB_normal/%s' % (OMPOSE_PATH, im_name.replace('png','csv')))
        flag = 0
        rects = []
        #img = cv.imread('%s/data/normal/images/%s'%(OMPOSE_PATH, im_name))
        #img = img[50:800, 650:1400]
        #cv.rectangle(img, (gt[0], gt[1]),(gt[2], gt[3]), (0,255,0),2)
        for eb in ebs:
            rect = []
            eb = np.asarray(eb.strip().split(',')[:4], dtype = int)
            if int(eb[0]) > int(eb[2]):
                lt_x = int(eb[2])
                rb_x = int(eb[0])
            else:
                lt_x = int(eb[0])
                rb_x = int(eb[2])
            if int(eb[1]) > int(eb[3]):
                lt_y = int(eb[3])
                rb_y = int(eb[1])
            else:
                lt_y = int(eb[1])
                rb_y = int(eb[3])
            eb = [lt_x,lt_y,rb_x,rb_y]
            IoU = eval_rects(gt, eb)
            if IoU > 0.1:
                rect.append(IoU)
                rect.append(eb[0])
                rect.append(eb[1])
                rect.append(eb[2])
                rect.append(eb[3])
                rects.append(rect)
                all_count += 1
            if IoU > 0.5:
                #cv.rectangle(img, (eb[0], eb[1]),(eb[2], eb[3]), (0,0,255),2)
                flag = 1
                p_count += 1
        if flag == 1:
            f = open('%s/data/normal/EB_samples/%s' % (OMPOSE_PATH, im_name.replace('png','csv')),'a')
            eb_writer = csv.writer(f)
            for x in rects:
                eb_writer.writerow(x)
    print('%d/%d'%(p_count, all_count))
    pbar.finish()
