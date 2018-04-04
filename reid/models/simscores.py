from __future__ import absolute_import
from torch import nn
from reid.models import CLASSIFIER
import torch
import torch.nn.init as init
import numpy as np

class SIMILARITY_SCORE(nn.Module):

    def __init__(self, class_num, input_num=128, drop=0):
        super(SIMILARITY_SCORE, self).__init__()

        self.classifier = CLASSIFIER(class_num, input_num, drop)


    def forward(self, input):
        inputsize = input.size()
        sample_num = inputsize[0]
        if sample_num % 2 != 0:
            raise RuntimeError("the batch size should be even number!")

        x = input.view(int(sample_num / 2), 2, -1)
        probe_x = x[:, 0, :]
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1, :]
        gallery_x = gallery_x.contiguous()


        cls_encode = self.classifier(probe_x, gallery_x)

        return cls_encode


def basicscore(**kwargs):
    return SIMILARITY_SCORE(2, **kwargs)