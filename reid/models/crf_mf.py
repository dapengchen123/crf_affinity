from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F





class CRF_MF(nn.Module):

    def __init__(self, Unarynum, Pairnum, layer_num=1):
        super(CRF_MF, self).__init__()
        self.Unarynum = Unarynum
        self.Pairnum = Pairnum
        self.layernum = layer_num
        self.weights = torch.nn.Parameter(torch.zeros(Unarynum+Pairnum))

    def forward(self, probescore, galleryscore):


        pairwise_mat = galleryscore - torch.diag(torch.diag(galleryscore))
        N = pairwise_mat.size()[0] - 1

        softmax_weights = F.softmax(self.weights)

        alphas = softmax_weights[0:self.Unarynum]
        betas = softmax_weights[self.Unarynum:self.Unarynum+self.Pairnum]

        # first consider one feature map
        norm_simsum = torch.sum(pairwise_mat, 0)
        norm_betassum = betas.expand_as(norm_simsum)
        norm_alphasum = alphas.expand_as(norm_simsum)
        normalizes = norm_alphasum + norm_betassum*norm_simsum

        mu = probescore
        mul_alpha = alphas.expand_as(mu)
        mul_beta  = betas.expand_as(mu)
        mul_normalizes = normalizes.expand_as(mu)
        for i in range(self.layernum):
            mu = (probescore * mul_alpha + mu.mm(pairwise_mat)* mul_beta)/mul_normalizes

        return mu




def crf_mf_1_1(**kwargs):
    return CRF_MF(1, 1, **kwargs)

