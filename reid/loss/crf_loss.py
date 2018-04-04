from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from reid.evaluator import accuracy


class CRFLoss(nn.Module):
    def __init__(self, sampling_rate=3):
        super(CRFLoss, self).__init__()
        self.sampling_rate = sampling_rate
        self.BCE = nn.BCELoss()
        self.BCE.size_average = False

    def forward(self, score, tar_probe, tar_gallery):

        cls_Size = score.size()
        N_probe = cls_Size[0]
        N_gallery = cls_Size[1]

        tar_gallery = tar_gallery.unsqueeze(0)
        tar_probe = tar_probe.unsqueeze(1)
        mask = tar_probe.expand(N_probe, N_gallery).eq(tar_gallery.expand(N_probe, N_gallery))
        mask = mask.view(-1).cpu().numpy().tolist()

        score = score.contiguous()
        samplers = score.view(-1)
        labels = Variable(torch.Tensor(mask).cuda())

        positivelabel = torch.Tensor(mask)
        negativelabel = 1 - positivelabel
        positiveweightsum = torch.sum(positivelabel)
        negativeweightsum = torch.sum(negativelabel)
        neg_relativeweight = positiveweightsum / negativeweightsum * self.sampling_rate
        weights = (positivelabel + negativelabel * neg_relativeweight)
        weights = weights/torch.sum(weights)/10
        self.BCE.weight = weights.cuda()
        loss = self.BCE(samplers, labels)

        samplers_data = samplers.data
        samplers_neg = 1 - samplers_data
        samplerdata = torch.cat((samplers_neg.unsqueeze(1),samplers_data.unsqueeze(1)), 1)

        labeldata = torch.LongTensor(mask).cuda()
        ## accuracy
        prec, = accuracy(samplerdata, labeldata)

        return loss, prec[0]