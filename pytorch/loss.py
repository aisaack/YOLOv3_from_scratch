import random
import numpy as np
import torch
import torch.nn as nn
from utils import iou

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, preds, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 1] == 0

        # no ojbect loss
        noobj_loss = self.bce(preds[..., 0:1][noobj], target[..., 0:1][noobj])

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)      # from (3, 2) shaped tensor
        box_preds = torch.cat(
                [
                    self.sigmoid(preds[..., 1:3]),             # b_x = sigmoid(t_x) + c_x 
                    torch.exp(preds[..., 3:5]) * anchors       # b_w = p_w * exp(t_w), where p_w is bbox prior
                ],
                dim = -1
        )
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        obj_loss = self.mse(self.sigmoid(preds[..., 0:1][obj]), ious * target[..., 0:1][obj])
        
        # bbox coordinate loss
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        target[...,3:5] = torch.log((1e-16 + target[...,3:5] / anchors))    # calculated from above equation, t_w = ln(b_w) / p_w 
        box_loss = self.mse(preds[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        cls_loss = self.entropy((preds[..., 5:][obj]), (target[..., 5:][obj]))

        return (self.lambda_box * box_loss
                + self.lambda_obj * obj_loss
                + self.lambda_noobj * noobj_loss
                + self.lambda_class * cls_loss)
