import random
import numpy as np
import torch
import torch.nn as nn
from utils import iou

from model import YoloLayer


def yolo_loss_fn(pred, tgt, model):
    device = tgt.device
    lcls, lbox, lobj = torch.zeros(1, device = device), torch.zeros(1, device = device), torch.zeros(1, device = device)
    tcls, tbox, indices, anchors = build_target(pred, tgt, model)

    for lyr_idx, lyr_pred in enumerate(pred):
        b, anchor, gi, gi = indices[lyr_idx]
        tobj = torch.zeros_like(lyr_pred[..., 0], device = device)
        nt = b.size(0)
        if nt:
            ps = lyr_pred[b, anchor, gj, gi]
            pxy = torch.sigmoid(ps[..., :2])
            pwh = torch.exp(ps[..., 2:4]) * anchors[lyr_idx]
            pbox = torch.cat([pxy, pwh], 1)
            iou = box_iou(pbox.T, tbox[lyr_idx], x1y1x2y2 = False)
            lbox += (1.0 - iou).meam()
            tobj[b, anchor, gj, gi] = 1

            if ps.size(1) - 5 > 1:
                t = torch.zeros_like(ps[..., 5:], device = device)
                t[range(nt), tcls[lyr_idx]] = 1
                lcls += F.bianry_cross_entropy_with_logits(lyr_pred[..., 4], tobj)

    lbox *= 0.05
    lcls *= 0.5
    lobj *= 1.0

    loss = lobx + lcls + lobj

    return loss, to_cpu(torch.cat([lbox, lobj, lcls, loss]))

def build_target(pred, tgt, model):
    na = 3
    nt = tgt.size(0)
    tcls, tbox, indices, anch = [], [], [], []
    ai = torch.arange(na, device = targets.device).float.view(na, 1).repeat(1, nt)
    targets = torch.cat([target.repeat(na, 1, 1), ai[:, None]], 2)
    gain = torch.ones(7)

    i = 0
    for m in moel.yolo_head.modules():
        if isinstance(m, YoloLayer):
            anchros = m.anchor / m.stride
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]
            i += 1
            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1./r).max(2)[0] < 4
                t = t[j]
            else:
                t = targets[0]
            
            b, c = t[..., :2].long().T
            gxy = t[..., 2:4]
            gwh = t[..., 4:6]
            gij = gxy.long()
            gi, gj = gij.T

            a = t[..., 6].long()
            indices.append((b, a, gj.clamp(0, (gain[3]-1).long()), gi.clamp(0, (gain[2]-1).long())))
            tbox.append(torch.cat([gxy - gij, gwh], 1))
            anch.append(anchor[a])
            tcls.append(c)

    return tcls, tbox, indices, anch

def bbox_iou(box1, box2, x1y1x2y2 = True, eps = 1e-9):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, boy1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, boy1[1] + box2[3] / 2

    inter = ((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
            * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0))

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    return inter / union


def build_target(pred, tgt, model):
    pass
