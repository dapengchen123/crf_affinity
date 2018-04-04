from __future__ import absolute_import

import torch
from torch import nn
from reid.loss import PairLoss


class MULPairLoss(nn.Module):

    def __init__(self, sampling_rate):
        super(MULPairLoss, self).__init__()
        self.sampling_rate = sampling_rate

        self.pairloss1 = PairLoss(self.sampling_rate)
        self.pairloss2 = PairLoss(self.sampling_rate)
        self.pairloss3 = PairLoss(self.sampling_rate)

    def forward(self, score1, score2, score3, tar_probe, tar_gallery):

        loss1, prec1 = self.pairloss1(score1, tar_probe, tar_gallery)
        loss2, prec2 = self.pairloss2(score2, tar_probe, tar_gallery)
        loss3, prec3 = self.pairloss3(score3, tar_probe, tar_gallery)

        loss = (loss1+loss2+loss3)/3
        prec = (prec1+prec2+prec3)/3

        return loss, prec