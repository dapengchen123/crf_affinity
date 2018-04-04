from __future__ import absolute_import
from torch import nn
from reid.models import  CLASSIFIER
import torch
import torch.nn.functional as F

class MULCLASSIFIER(nn.Module):

    def __init__(self, class_num, input_num=128, drop=0):
        super(MULCLASSIFIER, self).__init__()

        self.classifier1 = CLASSIFIER(class_num, input_num=input_num, drop=drop)
        self.classifier2 = CLASSIFIER(class_num, input_num=input_num, drop=drop)
        self.classifier3 = CLASSIFIER(class_num, input_num=input_num, drop=drop)

    def forward(self, probe_x1, gallery_x1, probe_x2, gallery_x2, probe_x3, gallery_x3):

        cls_encode1 = self.classifier1(probe_x1, gallery_x1)
        cls_encode2 = self.classifier2(probe_x2, gallery_x2)
        cls_encode3 = self.classifier3(probe_x3, gallery_x3)

        return cls_encode1, cls_encode2, cls_encode3


def multiclassifier2(**kwargs):
    return MULCLASSIFIER(2, **kwargs)
