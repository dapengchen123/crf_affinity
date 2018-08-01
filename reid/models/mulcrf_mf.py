from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F

class MULCRF_MF(nn.Module):

    def __init__(self, Unarynum, Pairnum, layer_num=3):
        super(MULCRF_MF, self).__init__()
        self.Unarynum = 3
        self.Pairnum = 3
        self.layernum = layer_num
        self.weights = torch.nn.Parameter(torch.ones(Unarynum+Pairnum))

    def forward(self, probescore1, galleryscore1,  probescore2, galleryscore2, probescore3, galleryscore3):

        pairwise_mat1 = galleryscore1 - torch.diag(torch.diag(galleryscore1))
        N1 = pairwise_mat1.size()[0] - 1
        pairwise_mat1 = pairwise_mat1/N1

        pairwise_mat2 = galleryscore2 - torch.diag(torch.diag(galleryscore2))
        N2 = pairwise_mat2.size()[0] - 1
        pairwise_mat2 = pairwise_mat2/N2

        pairwise_mat3 = galleryscore3 - torch.diag(torch.diag(galleryscore3))
        N3 = pairwise_mat3.size()[0] - 1
        pairwise_mat3 = pairwise_mat3/N3

        softmax_weights = F.softmax(self.weights, 0)

        alphas = softmax_weights[0:self.Unarynum]
        betas = softmax_weights[self.Unarynum:self.Unarynum + self.Pairnum]

        # first consider one feature map
        norm_simsum1 = torch.sum(pairwise_mat1, 0)
        norm_simsum2 = torch.sum(pairwise_mat2, 0)
        norm_simsum3 = torch.sum(pairwise_mat3, 0)
        beta1 = betas[0]
        beta2 = betas[1]
        beta3 = betas[2]
        norm_betassum1 = beta1.expand_as(norm_simsum1)
        norm_betassum2 = beta2.expand_as(norm_simsum2)
        norm_betassum3 = beta3.expand_as(norm_simsum3)

        alpha1 = alphas[0]
        alpha2 = alphas[1]
        alpha3 = alphas[2]

        norm_alphasum1 = alpha1.expand_as(norm_simsum1)
        norm_alphasum2 = alpha2.expand_as(norm_simsum2)
        norm_alphasum3 = alpha3.expand_as(norm_simsum3)

        normalizes = norm_alphasum1 + norm_alphasum2 + norm_alphasum3 + \
                     norm_betassum1 * norm_simsum1 + \
                     norm_betassum2 * norm_simsum2 + \
                     norm_betassum3 * norm_simsum3

        mu = (probescore1 + probescore2 + probescore3)/3

        mul_alpha1 = alpha1.expand_as(mu)
        mul_beta1 = beta1.expand_as(mu)

        mul_alpha2 = alpha2.expand_as(mu)
        mul_beta2 = beta2.expand_as(mu)

        mul_alpha3 = alpha3.expand_as(mu)
        mul_beta3 = beta3.expand_as(mu)

        mul_normalizes = normalizes.expand_as(mu)

        for i in range(self.layernum):
            mu = (probescore1 * mul_alpha1 + mu.mm(pairwise_mat1)*mul_beta1 +
                  probescore2 * mul_alpha2 + mu.mm(pairwise_mat2)*mul_beta2 +
                  probescore3 * mul_alpha3 + mu.mm(pairwise_mat3)*mul_beta3)/mul_normalizes

        return mu

def crf_mf_3_3(**kwargs):
    return MULCRF_MF(3, 3, **kwargs)
