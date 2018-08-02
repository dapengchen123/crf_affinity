from __future__ import absolute_import
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from reid.evaluator import cmc, mean_ap
from reid.utils.meters import AverageMeter
import time
import numpy as np

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
    print(distmat)
    distnp = distmat.cpu().numpy()
    np.save('dist_demo', distnp)

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
    print([cmc_scores['market1501'][k] for k in [0, 4, 9, 19]])

    ############### multiple query-images #######################

    joint_query = list(zip(query_ids, query_cams))
    summarize_index = list(set(joint_query))

    query_num = len(summarize_index)
    gallery_num = len(gallery_ids)
    sum_distmat = torch.zeros(query_num, gallery_num).cuda()
    for newind, sumind in enumerate(summarize_index):
        sum_indpos = [posx for posx, x in enumerate(joint_query) if x == sumind]
        ### mean results
        select_results = torch.index_select(distmat, 0 , torch.LongTensor(sum_indpos).cuda())
        select_mean = torch.mean(select_results, 0)
        sum_distmat[newind] = select_mean

    ## QUERY CAMERA

    multiquery_ids, multiquery_cams = zip(*summarize_index)
    multiquery_ids = list(multiquery_ids)
    multiquery_cams = list(multiquery_cams)

    multimAP = mean_ap(sum_distmat, multiquery_ids, gallery_ids, multiquery_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(multimAP))



    # Compute all kinds of CMC scores
    multicmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        # 'cuhk03': dict(separate_camera_set=True,
        #                single_gallery_shot=True,
        #                first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    multicmc_scores = {name: cmc(sum_distmat, multiquery_ids, gallery_ids,
                            multiquery_cams, gallery_cams, **params)
                  for name, params in multicmc_configs.items()}

    # print('CMC Scores{:>12}{:>12}{:>12}'
    #       .format('allshots', 'cuhk03', 'market1501'))
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #           .format(k, cmc_scores['allshots'][k - 1],
    #                   cmc_scores['cuhk03'][k - 1],
    #                   cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    print([multicmc_scores['market1501'][k] for k in [0, 4, 9, 19]])



    return [cmc_scores['market1501'][k] for k in [0, 4, 9, 19]]


class MsEvaluator(object):
    def __init__(self, cnnmodel, classifier, crfmodel):

        self.cnnmodel = cnnmodel
        self.classifier = classifier
        self.crfmodel = crfmodel

        softmax_weights = F.softmax(crfmodel.weights,0)
        self.alphas = softmax_weights[0:crfmodel.Unarynum]

    def extractfeature(self, data_loader):

        ## print
        print_freq = 10
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        queryfeat1 = 0
        queryfeat2 = 0
        queryfeat3 = 0
        preimgs = 0

        for i, (imgs, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                imgs = Variable(imgs)

                if i == 0:
                    query_feat1, query_feat2, query_feat3 = self.cnnmodel(imgs)
                    queryfeat1 = query_feat1
                    queryfeat2 = query_feat2
                    queryfeat3 = query_feat3
                    preimgs = imgs

                elif imgs.size(0) < data_loader.batch_size:

                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = data_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    query_feat1, query_feat2, query_feat3 = self.cnnmodel(imgs)

                    query_feat1 = query_feat1[0:flaw_batchsize]
                    query_feat2 = query_feat2[0:flaw_batchsize]
                    query_feat3 = query_feat3[0:flaw_batchsize]
                    queryfeat1 = torch.cat((queryfeat1, query_feat1), 0)
                    queryfeat2 = torch.cat((queryfeat2, query_feat2), 0)
                    queryfeat3 = torch.cat((queryfeat3, query_feat3), 0)
                else:
                    query_feat1, query_feat2, query_feat3 = self.cnnmodel(imgs)
                    queryfeat1 = torch.cat((queryfeat1, query_feat1), 0)
                    queryfeat2 = torch.cat((queryfeat2, query_feat2), 0)
                    queryfeat3 = torch.cat((queryfeat3, query_feat3), 0)

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return queryfeat1, queryfeat2, queryfeat3

    def evaluate(self, queryloader, galleryloader, query, gallery):
        softmax_weights = F.softmax(self.crfmodel.weights, 0)
        self.alphas = softmax_weights[0:self.crfmodel.Unarynum]
        print(self.alphas)
        distmat = self.compute_distmat(queryloader, galleryloader)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def compute_distmat(self, queryloader, galleryloader):
        self.cnnmodel.eval()
        self.classifier.eval()

        queryfeat1, queryfeat2, queryfeat3 = self.extractfeature(queryloader)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        print_freq = 50
        distmat = 0

        for i, (imgs, _, pids, _) in enumerate(galleryloader):
            data_time.update(time.time() - end)
            with torch.no_grad():
                imgs = Variable(imgs)

                if i == 0:
                    gallery_feat1, gallery_feat2, gallery_feat3 = self.cnnmodel(imgs)
                    preimgs = imgs
                elif imgs.size(0) < galleryloader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = galleryloader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    gallery_feat1, gallery_feat2, gallery_feat3 = self.cnnmodel(imgs)
                    gallery_feat1 = gallery_feat1[0:flaw_batchsize]
                    gallery_feat2 = gallery_feat2[0:flaw_batchsize]
                    gallery_feat3 = gallery_feat3[0:flaw_batchsize]
                else:
                    gallery_feat1, gallery_feat2, gallery_feat3 = self.cnnmodel(imgs)

                batch_cls_encode1, batch_cls_encode2, batch_cls_encode3 = self.classifier(queryfeat1, gallery_feat1,
                                                                                              queryfeat2,
                                                                                              gallery_feat2, queryfeat3,
                                                                                              gallery_feat3)
                batch_cls_size1 = batch_cls_encode1.size()
                batch_cls_encode1 = batch_cls_encode1.view(-1, 2)
                batch_cls_encode1 = F.softmax(batch_cls_encode1,1)
                batch_cls_encode1 = batch_cls_encode1.view(batch_cls_size1[0], batch_cls_size1[1], 2)
                batch_cls_encode1 = batch_cls_encode1[:, :, 0]

                batch_cls_size2 = batch_cls_encode2.size()
                batch_cls_encode2 = batch_cls_encode2.view(-1, 2)
                batch_cls_encode2 = F.softmax(batch_cls_encode2,1)
                batch_cls_encode2 = batch_cls_encode2.view(batch_cls_size2[0], batch_cls_size2[1], 2)
                batch_cls_encode2 = batch_cls_encode2[:, :, 0]

                batch_cls_size3 = batch_cls_encode3.size()
                batch_cls_encode3 = batch_cls_encode3.view(-1, 2)
                batch_cls_encode3 = F.softmax(batch_cls_encode3,1)
                batch_cls_encode3 = batch_cls_encode3.view(batch_cls_size3[0], batch_cls_size3[1], 2)
                batch_cls_encode3 = batch_cls_encode3[:, :, 0]

                batch_cls_encode = batch_cls_encode1 * self.alphas[0] + batch_cls_encode2 * self.alphas[1] + batch_cls_encode3 * self.alphas[2]
                if i == 0:
                    distmat = batch_cls_encode.data
                else:
                    distmat = torch.cat((distmat, batch_cls_encode.data), 1)

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Data {:.3f} ({:.3f})\t'
                              .format(i + 1, len(galleryloader),
                                      batch_time.val, batch_time.avg,
                                      data_time.val, data_time.avg))
        return distmat
