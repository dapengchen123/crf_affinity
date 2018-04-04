from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from torch.autograd import Variable
import numpy as np

from reid.evaluator import cmc, mean_ap
from reid.utils.meters import AverageMeter
import torch.nn.functional as F



def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

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


class BinaryEvaluator(object):
    def __init__(self, cnnmodel, scoremodel):
        super(BinaryEvaluator, self).__init__()
        self.cnnmodel = cnnmodel
        self.classifier = scoremodel.classifier

    def extractfeature(self, data_loader):
        ## print
        print_freq = 50
        self.cnnmodel.eval()
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

    def evaluate(self, queryloader, galleryloader,  query, gallery):

        query_features = self.extractfeature(queryloader)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        print_freq = 50
        distmat = 0

        self.cnnmodel.eval()
        self.classifier.eval()

        for i, (imgs,_ , pids, _) in enumerate(galleryloader):
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
            batch_encode = batch_cls_encode[:, :, 0]

            if i == 0:
                distmat = batch_encode.data
            else:
                distmat = torch.cat((distmat, batch_encode.data), 1)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(galleryloader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))


        return evaluate_all(distmat, query=query, gallery=gallery)