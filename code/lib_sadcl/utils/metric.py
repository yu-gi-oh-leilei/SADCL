# most borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py

import numpy as np
import torch
import math
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False, dataname='wider'):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32) # label
    num_target = np.sum(gt_label, axis=1, keepdims = True) # output
    
    # all_threshold = 0.59
    # top3_threshold = 0.59
    # overall(seg[:,:num] ,gt_label, all_threshold)
    # overall_topk(seg[:,:num] ,gt_label, top3_threshold, 3)

    # all_threshold = 0.39
    # top3_threshold = 0.39
    # for i in range(30):
    #     print(all_threshold + 0.01, top3_threshold + 0.01)
    #     all_threshold = all_threshold +  0.01
    #     top3_threshold = top3_threshold +  0.01
    #     overall(seg[:,:num] ,gt_label, all_threshold)
    #     overall_topk(seg[:,:num] ,gt_label, top3_threshold, 3)
    #     print('==='*20)
    # return 

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []



    for class_id in range(class_num):
        confidence = seg[:,class_id]


        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]


        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP


def overall_topk(output, gt_label, threshold, k=3):    
    output_ = torch.from_numpy(output).clone()
    idx = torch.sort(-output_)[1]
    idx_after_k = idx[:,k:]
    output_.scatter_(1, idx_after_k,0.)
    output_ = output_.gt(threshold).long()

    tp = (output_ + torch.from_numpy(gt_label).clone()).eq(2).sum(dim=0)
    fp = (output_ - torch.from_numpy(gt_label).clone()).eq(1).sum(dim=0)
    fn = (output_ - torch.from_numpy(gt_label).clone()).eq(-1).sum(dim=0)
    tn = (output_ + torch.from_numpy(gt_label).clone()).eq(0).sum(dim=0)

    c_p = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    c_r = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    c_f = [2 * c_p[i] * c_r[i] / (c_p[i] + c_r[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

    mean_c_p = sum(c_p) / len(c_p)
    mean_c_r = sum(c_r) / len(c_r)
    mean_c_f = sum(c_f) / len(c_f)

    o_p = tp.sum().float() / (tp + fp).sum().float() * 100.0
    o_r = tp.sum().float() / (tp + fn).sum().float() * 100.0
    o_f = 2 * o_p * o_r / (o_p + o_r)
    # print(' * CP3 {:.2f} CR3 {:.2f} CF3 {:.2f} OP3 {:.2f} OR3 {:.2f} OF3 {:.2f}'.format(mean_c_p, mean_c_r, mean_c_f, o_p, o_r, o_f))
    print(' *   CP3    CR3    CF3    OP3    OR3    OF3 ')
    print(' *  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}'.format(mean_c_p, mean_c_r, mean_c_f, o_p, o_r, o_f))



def overall(output, gt_label, threshold):
    # pred = torch.from_numpy(seg[:,:num]).data.gt(0.83).long()
    pred = torch.from_numpy(output).clone().data.gt(threshold).long()
    # print(pred)
    tp = (pred + torch.from_numpy(gt_label).clone()).eq(2).sum(dim=0)
    fp = (pred - torch.from_numpy(gt_label).clone()).eq(1).sum(dim=0)
    fn = (pred - torch.from_numpy(gt_label).clone()).eq(-1).sum(dim=0)
    tn = (pred + torch.from_numpy(gt_label).clone()).eq(0).sum(dim=0)

    c_p = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    c_r = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))]
    c_f = [2 * c_p[i] * c_r[i] / (c_p[i] + c_r[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]

    mean_c_p = sum(c_p) / len(c_p)
    mean_c_r = sum(c_r) / len(c_r)
    mean_c_f = sum(c_f) / len(c_f)

    o_p = tp.sum().float() / (tp + fp).sum().float() * 100.0
    o_r = tp.sum().float() / (tp + fn).sum().float() * 100.0
    o_f = 2 * o_p * o_r / (o_p + o_r)
    # print(' * CP {:.2f} CR {:.2f} CF {:.2f} OP {:.2f} OR {:.2f} OF {:.2f}'.format(mean_c_p, mean_c_r, mean_c_f, o_p, o_r, o_f))
    print(' *    CP     CR     CF     OP     OR     OF ')
    print(' *  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}'.format(mean_c_p, mean_c_r, mean_c_f, o_p, o_r, o_f))


