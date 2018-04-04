from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from reid.evaluator import accuracy


class PairwiseLoss(nn.Module):
    def __init__(self, sampling_rate, batchsize):
        super(PairwiseLoss, self).__init__()
        self.sampling_rate = sampling_rate
        weights = torch.Tensor([sampling_rate/(batchsize-1), 1])
        self.crossentropy = nn.CrossEntropyLoss(weights)


    def forward(self, cls_encode, tar_probe, tar_gallery):

        cls_Size = cls_encode.size()
        N_probe = cls_Size[0]
        N_gallery = cls_Size[1]
        tar_gallery = tar_gallery.unsqueeze(0)
        tar_probe = tar_probe.unsqueeze(1)
        mask = tar_probe.expand(N_probe, N_gallery).eq(tar_gallery.expand(N_probe, N_gallery))
        mask = mask.view(-1).cpu().numpy().tolist()

        samplers = cls_encode.view(-1, 2)
        labels = Variable(torch.LongTensor(mask).cuda())
        loss = self.crossentropy(samplers, labels)
        ## accuracy
        prec, = accuracy(samplers.data, labels.data)

        return loss, prec[0]

