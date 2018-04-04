from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd
from reid.loss import OIMLoss

class MULOIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0):
        super(MULOIMLoss, self).__init__()
        self.oimloss1 = OIMLoss(num_features, num_classes, scalar=scalar, momentum=momentum)
        self.oimloss2 = OIMLoss(num_features, num_classes, scalar=scalar, momentum=momentum)
        self.oimloss3 = OIMLoss(num_features, num_classes, scalar=scalar, momentum=momentum)

    def forward(self, feature1, feature2, feature3, targets):

        loss1, outputs1 = self.oimloss1(feature1, targets)
        loss2, outputs2 = self.oimloss2(feature2, targets)
        loss3, outputs3 = self.oimloss3(feature3, targets)

        loss = (loss1 + loss2 + loss3)/3

        return loss, outputs1, outputs2, outputs3
