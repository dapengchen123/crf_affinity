from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from reid.evaluator import accuracy
from reid.loss import OIMLoss, TripletLoss
from reid.utils.meters import AverageMeter
from torch.nn import functional as F


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        precisions2 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)

            loss, prec_oim, loss_score, prec_finalscore = self._forward(inputs, targets, i)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec_oim, targets.size(0))
            precisions1.update(loss_score.data[0], targets.size(0))
            precisions2.update(prec_finalscore, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 50
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'prec_oim {:.2%} ({:.2%})\t'
                      'prec_score {:.2%} ({:.2%})\t'
                      'prec_finalscore(total) {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              precisions1.val, precisions1.avg,
                              precisions2.val, precisions2.avg))


    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class BinaryTrainer(BaseTrainer):

    def __init__(self, cnnmodel, scoremodel, criterion):
        super(BinaryTrainer, self).__init__(cnnmodel, criterion)
        self.scoremodel = scoremodel

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):

        out_feat = self.model(inputs[0])
        featsize = out_feat.size()
        sample_num = featsize[0]

        scores = self.scoremodel(out_feat)
        targets = targets.data
        targets = targets.view(int(sample_num / 2), -1)
        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1]

        loss, prec = self.criterion(scores, tar_probe, tar_gallery)
        return loss, prec

    def train(self, epoch, data_loader, optimizer):
        self.scoremodel.train()
        super(BinaryTrainer, self).train(epoch, data_loader, optimizer)



class JointTrainer(BaseTrainer):

    def __init__(self, cnnmodel, scoremodel, criterion1, criterion2, rate=0.1):
        super(JointTrainer, self).__init__(cnnmodel, criterion1)
        self.scoremodel = scoremodel
        self.regular_criterion = criterion2
        self.rate1 = rate
        self.rate2 = 1 -rate
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):

        out_feat = self.model(inputs[0])
        featsize = out_feat.size()
        sample_num = featsize[0]

        ## identification
        loss1, outputs =  self.regular_criterion(out_feat, targets)
        prec1, = accuracy(outputs.data, targets.data)
        prec1  = prec1[0]

        ## verification
        scores = self.scoremodel(out_feat)
        targets = targets.data
        targets = targets.view(int(sample_num / 2), -1)
        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1]

        loss2, prec2 = self.criterion(scores, tar_probe, tar_gallery)


        loss = loss1*self.rate1 + loss2*self.rate2
        prec3 = prec1*self.rate1 + prec2*self.rate2

        return loss, prec3

    def train(self, epoch, data_loader, optimizer):
        self.scoremodel.train()
        super(JointTrainer, self).train(epoch, data_loader, optimizer)




class CRFTrainer(BaseTrainer):

    def __init__(self, cnnmodel, classifier, crf_mf, criterion_oim, criterion_score, criterion_crf, instances_num):
        super(CRFTrainer, self).__init__(cnnmodel, criterion_crf)
        self.classifier = classifier
        self.crf_mf = crf_mf
        self.regular_criterion = criterion_oim
        self.criterion_score = criterion_score
        self.instances_num = instances_num


    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):

        out_feat = self.model(inputs[0])
        featsize = out_feat.size()
        sample_num = featsize[0]
        feat_num = featsize[1]

        ## identification
        loss_oim, outputs = self.regular_criterion(out_feat, targets)
        prec_oim, = accuracy(outputs.data, targets.data)
        prec_oim  = prec_oim[0]

        ## verification
        ### labels

        targets = targets.data
        targets = targets.view(int(sample_num / self.instances_num), -1)

        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1:self.instances_num]
        tar_gallery = tar_gallery.contiguous()
        tar_gallery = tar_gallery.view(-1)


        ### features
        x = out_feat.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x = x[:, 0, :]
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1:self.instances_num, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(-1, feat_num)


        encode_scores =   self.classifier(probe_x, gallery_x)
        gallery_scores  = self.classifier(gallery_x, gallery_x)
        encode_size = encode_scores.size()

        encodemat = encode_scores.view(-1, 2)
        encodemat = F.softmax(encodemat,1)
        encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
        initialscore = encodemat[:, :, 1]

        loss_score, prec_score = self.criterion_score(initialscore, tar_probe, tar_gallery)

        gallery_size = gallery_scores.size()
        gallerymat = gallery_scores.view(-1, 2)
        gallerymat = F.softmax(gallerymat,1)
        gallerymat = gallerymat.view(gallery_size[0], gallery_size[1], 2)
        gallerymat = gallerymat[:, :, 1]

        finalscores = self.crf_mf(initialscore, gallerymat)
        loss_finalscore, prec_finalscore = self.criterion(finalscores, tar_probe, tar_gallery)

        loss = loss_oim*0.2 + loss_score*0.4 + loss_finalscore*0.4


        return loss, prec_oim, prec_score, prec_finalscore

    def train(self, epoch, data_loader, optimizer):
        self.classifier.train()
        self.crf_mf.train()
        super(CRFTrainer, self).train(epoch, data_loader, optimizer)
        print(F.softmax(self.crf_mf.weights))


class NOCRFTrainer(BaseTrainer):

    def __init__(self, cnnmodel, classifier, verifi_criterion, oim_criterion, instances_num):
        super(NOCRFTrainer, self).__init__(cnnmodel, verifi_criterion)
        self.classifier = classifier
        self.regular_criterion = oim_criterion
        self.instances_num = instances_num


    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):

        out_feat = self.model(inputs[0])
        featsize = out_feat.size()
        sample_num = featsize[0]
        feat_num = featsize[1]

        ## identification
        loss1, outputs = self.regular_criterion(out_feat, targets)
        prec1, = accuracy(outputs.data, targets.data)
        prec1  = prec1[0]

        ## verification
        ### labels

        targets = targets.data
        targets = targets.view(int(sample_num / self.instances_num), -1)

        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1:self.instances_num]
        tar_gallery = tar_gallery.contiguous()
        tar_gallery = tar_gallery.view(-1)


        ### features
        x = out_feat.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x = x[:, 0, :]
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1:self.instances_num, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(-1, feat_num)


        encode_scores =   self.classifier(probe_x, gallery_x)
        encode_size = encode_scores.size()
        encodemat = encode_scores.view(-1, 2)
        encodemat = F.softmax(encodemat)
        encodemat = encodemat.view(encode_size[0], encode_size[1], 2)
        encodemat = encodemat[:, :, 1]

        loss2, prec2 = self.criterion(encodemat, tar_probe, tar_gallery)

        loss = loss1*0.2+ loss2*0.8
        prec3 = prec1*0.2 + prec2*0.8

        return loss, prec1, prec2, prec3

    def train(self, epoch, data_loader, optimizer):
        self.classifier.train()
        super(NOCRFTrainer, self).train(epoch, data_loader, optimizer)


class MULJOINTTrainer(BaseTrainer):

    def __init__(self, cnnmodel, mulclassifier, crfmf, verifi_criterion, oim_criterion, instances_num):
        super(MULJOINTTrainer, self).__init__(cnnmodel, verifi_criterion)
        self.mulclassifier = mulclassifier
        self.regular_criterion = oim_criterion
        self.instances_num = instances_num
        self.crf_mf = crfmf

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):

        featlayer4, featlayer3, featlayer2 = self.model(inputs[0])

        ## identification
        loss_id, outputs4, outputs3, outputs2 = self.regular_criterion(featlayer4, featlayer3, featlayer2, targets)

        precid4, = accuracy(outputs4.data, targets.data)
        precid4 = precid4[0]

        precid3, = accuracy(outputs3.data, targets.data)
        precid3 = precid3[0]

        precid2, = accuracy(outputs2.data, targets.data)
        precid2 = precid2[0]

        precid = (precid2 + precid3 + precid4)/3

        featsize = featlayer4.size()
        sample_num = featsize[0]
        feat_num = featsize[1]
        ## verification
        ### labels

        targets = targets.data
        targets = targets.view(int(sample_num / self.instances_num), -1)

        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1:self.instances_num]
        tar_gallery = tar_gallery.contiguous()
        tar_gallery = tar_gallery.view(-1)
        ### features
        x4 = featlayer4.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x4 = x4[:, 0, :]
        probe_x4 = probe_x4.contiguous()
        gallery_x4 = x4[:, 1:self.instances_num, :]
        gallery_x4 = gallery_x4.contiguous()
        gallery_x4 = gallery_x4.view(-1, feat_num)

        x3 = featlayer3.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x3 = x3[:, 0, :]
        probe_x3 = probe_x3.contiguous()
        gallery_x3 = x3[:, 1:self.instances_num, :]
        gallery_x3 = gallery_x3.contiguous()
        gallery_x3 = gallery_x3.view(-1, feat_num)

        x2 = featlayer2.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x2 = x2[:, 0, :]
        probe_x2 = probe_x2.contiguous()
        gallery_x2 = x2[:, 1:self.instances_num, :]
        gallery_x2 = gallery_x2.contiguous()
        gallery_x2 = gallery_x2.view(-1, feat_num)



        encode_scores4, encode_scores3, encode_scores2 = self.mulclassifier(probe_x4, gallery_x4, probe_x3, gallery_x3,
                                                                            probe_x2, gallery_x2)


        encode_size4 = encode_scores4.size()
        encodemat4 = encode_scores4.view(-1, 2)
        encodemat4 = F.softmax(encodemat4, 1)
        encodemat4 = encodemat4.view(encode_size4[0], encode_size4[1], 2)
        encodemat4 = encodemat4[:, :, 1]

        encode_size3 = encode_scores3.size()
        encodemat3 = encode_scores3.view(-1, 2)
        encodemat3 = F.softmax(encodemat3, 1)
        encodemat3 = encodemat3.view(encode_size3[0], encode_size3[1], 2)
        encodemat3 = encodemat3[:, :, 1]

        encode_size2 = encode_scores2.size()
        encodemat2 = encode_scores2.view(-1, 2)
        encodemat2 = F.softmax(encodemat2, 1)
        encodemat2 = encodemat2.view(encode_size2[0], encode_size2[1], 2)
        encodemat2 = encodemat2[:, :, 1]

        ## joint learning
        #encodemat = (encodemat2 + encodemat3 + encodemat4) / 3


        ## joint learning
        #loss_veri, prec_veri = self.criterion(encodemat, tar_probe, tar_gallery)

        ## CRF loss

        gallery_scores4, gallery_scores3, gallery_scores2 = self.mulclassifier(gallery_x4, gallery_x4, gallery_x3, gallery_x3, gallery_x2, gallery_x2)

        gallery_size4 = gallery_scores4.size()
        gallerymat4 = gallery_scores4.view(-1, 2)
        gallerymat4 = F.softmax(gallerymat4, 1)
        gallerymat4 = gallerymat4.view(gallery_size4[0], gallery_size4[1], 2)
        gallerymat4 = gallerymat4[:, :, 1]

        gallery_size3 = gallery_scores3.size()
        gallerymat3 = gallery_scores3.view(-1, 2)
        gallerymat3 = F.softmax(gallerymat3, 1)
        gallerymat3 = gallerymat3.view(gallery_size3[0], gallery_size3[1], 2)
        gallerymat3 = gallerymat3[:, :, 1]

        gallery_size2 = gallery_scores2.size()
        gallerymat2 = gallery_scores2.view(-1, 2)
        gallerymat2 = F.softmax(gallerymat2, 1)
        gallerymat2 = gallerymat2.view(gallery_size2[0], gallery_size2[1], 2)
        gallerymat2 = gallerymat2[:, :, 1]

        globalscores = self.crf_mf(encodemat4, gallerymat4, encodemat3, gallerymat3, encodemat2, gallerymat2)


        loss_global, prec_global = self.criterion(globalscores, tar_probe, tar_gallery)

        loss = loss_id*0.5 + loss_global

        return loss, precid, 0, prec_global

    def train(self, epoch, data_loader, optimizer):
        self.mulclassifier.train()
        self.crf_mf.train()
        super(MULJOINTTrainer, self).train(epoch, data_loader, optimizer)
    

class MULJOINT_MAN_Trainer(BaseTrainer):

    def __init__(self, cnnmodel, mulclassifier, crfmf, verifi_criterion, oim_criterion, instances_num):
        super(MULJOINT_MAN_Trainer, self).__init__(cnnmodel, verifi_criterion)
        self.mulclassifier = mulclassifier
        self.regular_criterion = oim_criterion
        self.instances_num = instances_num
        self.crf_mf = crfmf

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def pairwise_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)

        dist = torch.pow(x, 2).sum(1).expand(m, n) + \
               torch.pow(y, 2).sum(1).expand(n, m).t()
        
        dist.addmm_(1, -2, x, y.t())
        dist = torch.abs(dist)

        return dist
    
    
    def _forward(self, inputs, targets, it):

        featlayer4, featlayer3, featlayer2 = self.model(inputs[0])

        ## identification
        loss_id, outputs4, outputs3, outputs2 = self.regular_criterion(featlayer4, featlayer3, featlayer2, targets)

        precid4, = accuracy(outputs4.data, targets.data)
        precid4 = precid4[0]

        precid3, = accuracy(outputs3.data, targets.data)
        precid3 = precid3[0]

        precid2, = accuracy(outputs2.data, targets.data)
        precid2 = precid2[0]

        precid = (precid2 + precid3 + precid4)/3

        featsize = featlayer4.size()
        sample_num = featsize[0]
        feat_num = featsize[1]
        ## verification
        ### labels

        targets = targets.data
        targets = targets.view(int(sample_num / self.instances_num), -1)

        tar_probe = targets[:, 0]
        tar_gallery = targets[:, 1:self.instances_num]
        tar_gallery = tar_gallery.contiguous()
        tar_gallery = tar_gallery.view(-1)
        ### features
        x4 = featlayer4.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x4 = x4[:, 0, :]
        probe_x4 = probe_x4.contiguous()
        gallery_x4 = x4[:, 1:self.instances_num, :]
        gallery_x4 = gallery_x4.contiguous()
        gallery_x4 = gallery_x4.view(-1, feat_num)

        x3 = featlayer3.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x3 = x3[:, 0, :]
        probe_x3 = probe_x3.contiguous()
        gallery_x3 = x3[:, 1:self.instances_num, :]
        gallery_x3 = gallery_x3.contiguous()
        gallery_x3 = gallery_x3.view(-1, feat_num)

        x2 = featlayer2.view(int(sample_num / self.instances_num), self.instances_num, -1)
        probe_x2 = x2[:, 0, :]
        probe_x2 = probe_x2.contiguous()
        gallery_x2 = x2[:, 1:self.instances_num, :]
        gallery_x2 = gallery_x2.contiguous()
        gallery_x2 = gallery_x2.view(-1, feat_num)



        encode_scores4, encode_scores3, encode_scores2 = self.mulclassifier(probe_x4, gallery_x4, probe_x3, gallery_x3,
                                                                            probe_x2, gallery_x2)


        encode_size4 = encode_scores4.size()
        encodemat4 = encode_scores4.view(-1, 2)
        encodemat4 = F.softmax(encodemat4, 1)
        encodemat4 = encodemat4.view(encode_size4[0], encode_size4[1], 2)
        encodemat4 = encodemat4[:, :, 1]

        encode_size3 = encode_scores3.size()
        encodemat3 = encode_scores3.view(-1, 2)
        encodemat3 = F.softmax(encodemat3, 1)
        encodemat3 = encodemat3.view(encode_size3[0], encode_size3[1], 2)
        encodemat3 = encodemat3[:, :, 1]

        encode_size2 = encode_scores2.size()
        encodemat2 = encode_scores2.view(-1, 2)
        encodemat2 = F.softmax(encodemat2, 1)
        encodemat2 = encodemat2.view(encode_size2[0], encode_size2[1], 2)
        encodemat2 = encodemat2[:, :, 1]

        ## joint learning
        #encodemat = (encodemat2 + encodemat3 + encodemat4) / 3


        ## joint learning
        #loss_veri, prec_veri = self.criterion(encodemat, tar_probe, tar_gallery)

        ## CRF loss
        
        gallery_x4_data = Variable(gallery_x4.data, requires_grad=False)
        gallery_x3_data = Variable(gallery_x3.data, requires_grad=False)
        gallery_x2_data = Variable(gallery_x2.data, requires_grad=False)


        gamma=3
        
        gallery_featdist4 = self.pairwise_dist(gallery_x4_data, gallery_x4_data)
        gallery_featdist3 = self.pairwise_dist(gallery_x3_data, gallery_x3_data)
        gallery_featdist2 = self.pairwise_dist(gallery_x2_data, gallery_x2_data)
    

        
        gallery_scores4 = torch.exp(-gallery_featdist4*gamma)
        gallery_scores3 = torch.exp(-gallery_featdist3*gamma)
        gallery_scores2 = torch.exp(-gallery_featdist2*gamma)
        
        if it==1:
            print("gallery scores")
            print(gallery_scores4)
        
        globalscores = self.crf_mf(encodemat4, gallery_scores4, encodemat3, gallery_scores3, encodemat2, gallery_scores2)


        loss_global, prec_global = self.criterion(globalscores, tar_probe, tar_gallery)

        loss = loss_id + loss_global*50

        return loss, precid, loss_global, prec_global

    def train(self, epoch, data_loader, optimizer):
        self.mulclassifier.train()
        self.crf_mf.train()
        super(MULJOINT_MAN_Trainer, self).train(epoch, data_loader, optimizer)

   
