from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from torch.autograd import Variable
import numpy as np
from reid.evaluator import cmc, mean_ap
from reid.feature_extraction import extract_cnn_feature
from reid.utils.meters import AverageMeter
from reid.utils import to_torch
import torch.nn.functional as F


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


class PartialCRFEvaluator(object):
    def __init__(self, cnnmodel, classifier, crfmodel):
        self.cnnmodel = cnnmodel
        self.classifier = classifier
        self.crfmodel = crfmodel
        self.select_size = 50

    def evaluate_1st(self, distmat, query=None, gallery=None):
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
        else:
            raise RuntimeError('please provide the query and gallery information')
        distmat = to_numpy(distmat)
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        print('Mean AP: {:4.1%}'.format(mAP))

        indices = np.argsort(distmat, axis=1)
        indices = np.argsort(indices, axis=1)
        mask = (indices < self.select_size).astype(float)

        cmc_configs = {
            # 'allshots': dict(separate_camera_set=False,
            #                  single_gallery_shot=False,
            #                  first_match_break=False),
            # 'cuhk03': dict(separate_camera_set=True,
            #                single_gallery_shot=True,
            #                first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        return [cmc_scores['market1501'][k] for k in [0, 4, 9, 19]], mask

    def evaluate_all(self, distmat, query=None, gallery=None):
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
        else:
            raise RuntimeError('please provide the query and gallery information')


        # Compute mean AP
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        print('Mean AP: {:4.1%}'.format(mAP))

        # Compute all kinds of CMC scores
        cmc_configs = {
            # 'allshots': dict(separate_camera_set=False,
            #                  single_gallery_shot=False,
            #                  first_match_break=False),
            # 'cuhk03': dict(separate_camera_set=True,
            #                single_gallery_shot=True,
            #                first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        # print('CMC Scores{:>12}{:>12}{:>12}'
        #       .format('allshots', 'cuhk03', 'market1501'))
        # for k in cmc_topk:
        #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
        #           .format(k, cmc_scores['allshots'][k - 1],
        #                   cmc_scores['cuhk03'][k - 1],
        #                   cmc_scores['market1501'][k - 1]))

        # Use the allshots cmc top-1 score for validation criterion
        return [cmc_scores['market1501'][k] for k in [0, 4, 9, 19]]

    def extractfeature(self, data_loader):
        ## print
        print_freq = 50
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        queryfeatures = 0
        preimgs = 0
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = Variable(imgs, volatile=True)
            if i == 0:
                query_feat = self.cnnmodel(imgs)
                queryfeatures = query_feat
                preimgs = imgs
            elif imgs.size(0) < data_loader.batch_size:
                flaw_batchsize = imgs.size(0)
                cat_batchsize = data_loader.batch_size - flaw_batchsize
                imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                query_feat = self.cnnmodel(imgs)
                query_feat = query_feat[0:flaw_batchsize]
                queryfeatures = torch.cat((queryfeatures, query_feat), 0)
            else:
                query_feat = self.cnnmodel(imgs)
                queryfeatures = torch.cat((queryfeatures, query_feat), 0)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
        return queryfeatures

    def sim_computation(self, galleryloader, query_features):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        print_freq = 50
        simmat = 0

        for i, (imgs, _, pids, _) in enumerate(galleryloader):
            data_time.update(time.time() - end)
            imgs = Variable(imgs, volatile=True)

            if i == 0:
                gallery_feat = self.cnnmodel(imgs)
                preimgs = imgs

            elif imgs.size(0) < galleryloader.batch_size:

                flaw_batchsize = imgs.size(0)
                cat_batchsize = galleryloader.batch_size - flaw_batchsize
                imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                gallery_feat = self.cnnmodel(imgs)
                gallery_feat = gallery_feat[0:flaw_batchsize]

            else:
                gallery_feat = self.cnnmodel(imgs)


            batch_cls_encode = self.classifier(query_features, gallery_feat)
            batch_cls_size = batch_cls_encode.size()
            batch_cls_encode = batch_cls_encode.view(-1, 2)
            batch_cls_encode = F.softmax(batch_cls_encode)
            batch_cls_encode = batch_cls_encode.view(batch_cls_size[0], batch_cls_size[1], 2)
            batch_similarity = batch_cls_encode[:, :, 1]

            if i == 0:
                simmat = batch_similarity
            else:
                simmat = torch.cat((simmat, batch_similarity), 1)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(galleryloader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        return simmat

    def partial_crf(self, probescore, galleryscore, mask):
        ##  换成 numpy 进行计算
        ##   probescore, galleryscore torch
        ##   mask numpy nd arrary

        pairwise_mat = galleryscore - np.diag(np.diag(galleryscore))
        pairwise_mat = pairwise_mat / (self.select_size-1)


        softmax_weights = F.softmax(self.crfmodel.weights).data
        alphas = softmax_weights[0:self.crfmodel.Unarynum].cpu().numpy()
        betas = softmax_weights[self.crfmodel.Unarynum:self.crfmodel.Unarynum + self.crfmodel.Pairnum].cpu().numpy()


        norm_simsum = np.dot(mask, pairwise_mat)
        normalizes = alphas + norm_simsum*betas

        mu = probescore * mask
        for i in range(self.crfmodel.layernum):
            mu = (probescore * alphas + np.dot(mu, pairwise_mat*betas)) / normalizes
            mu = mu * mask
        return mu


    def evaluate(self, queryloader, galleryloader, query, gallery):
        self.cnnmodel.eval()
        self.classifier.eval()
        self.crfmodel.eval()



        query_features = self.extractfeature(queryloader)
        # gallery_features = self.extractfeature(galleryloader)
        simmat = self.sim_computation(galleryloader, query_features)

        # top0, mask = self.evaluate_1st(1 - simmat.data, query = query, gallery = gallery)
        # print(top0)
        # gallerymat = self.sim_computation(galleryloader, gallery_features)
        # ### partial crf model
        # simmat = simmat.data.cpu().numpy()
        # gallerymat = gallerymat.data.cpu().numpy()
        # print(np.amax(simmat))
        # print(np.amin(simmat))
        #
        # scores = self.partial_crf(simmat, gallerymat, mask)
        # final_scores = (scores+1)*mask + simmat*(1-mask)
        # print(np.amax(final_scores))
        # print(np.amin(final_scores))

        final_scores = simmat.data.cpu().numpy()


        return self.evaluate_all(2-final_scores, query = query, gallery = gallery)






