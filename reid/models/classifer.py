from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
import numpy as np


class CLASSIFIER(nn.Module):

    def __init__(self, class_num, input_num=128, drop=0):
        super(CLASSIFIER, self).__init__()
        self.feat_num = input_num
        self.class_num = class_num

        ## BN layer
        self.featKC_bn = nn.BatchNorm1d(self.feat_num)

        ## Classifier
        self.classifier = nn.Linear(self.feat_num, self.class_num)

        ## dropout layer
        self.drop = drop

        if self.drop > 0:
            self.droplayer = nn.Dropout(drop)

        init.constant(self.featKC_bn.weight, 1)
        init.constant(self.featKC_bn.bias, 0)

        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, probe_x, gallery_x):
        S_probe = probe_x.size()
        N_probe = S_probe[0]
        S_gallery = gallery_x.size()
        N_gallery = S_gallery[0]

        probe_x = probe_x.unsqueeze(1)
        probe_x = probe_x.expand(N_probe, N_gallery, self.feat_num)
        probe_x = probe_x.contiguous()

        gallery_x = gallery_x.unsqueeze(0)
        gallery_x = gallery_x.expand(N_probe, N_gallery, self.feat_num)
        gallery_x = gallery_x.contiguous()

        diff = torch.pow(probe_x - gallery_x, 2)
        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()
        slice = 10000
        if N_probe * N_gallery < slice:
            diff = self.featKC_bn(diff)
            if self.drop > 0:
                diff = self.droplayer(diff)

            cls_encode = self.classifier(diff)
            cls_encode = cls_encode.view(N_probe, N_gallery, -1)

        else:

            Iter_time = int(np.floor(N_probe * N_gallery / slice))
            cls_encode = 0
            for i in range(0, Iter_time):
                before_index = i * slice
                after_index = (i + 1) * slice

                diff_i = diff[before_index:after_index, :]
                diff_i = self.featKC_bn(diff_i)

                if self.drop > 0:
                    diff_i = self.droplayer(diff_i)

                cls_encode_i = self.classifier(diff_i)

                if i == 0:
                    cls_encode = cls_encode_i
                else:
                    cls_encode = torch.cat((cls_encode, cls_encode_i), 0)

            before_index = Iter_time * slice
            after_index = N_probe * N_gallery
            if after_index > before_index:
                diff_i = diff[before_index:after_index, :]
                diff_i = self.featKC_bn(diff_i)
                if self.drop > 0:
                    diff_i = self.droplayer(diff_i)

                cls_encode_i = self.classifier(diff_i)
                cls_encode = torch.cat((cls_encode, cls_encode_i), 0)
            cls_encode = cls_encode.view(N_probe, N_gallery, self.class_num)

        return cls_encode


def classifier2(**kwargs):
    return CLASSIFIER(2, **kwargs)
