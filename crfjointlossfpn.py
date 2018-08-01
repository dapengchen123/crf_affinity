import argparse
import os.path as osp
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn

from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.data import get_data
from reid import models
from reid.loss import PairLoss
from reid.loss import MULOIMLoss
from reid.train import MULJOINT_MAN_Trainer
from reid.evaluator import MsEvaluator


from reid import datasets


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create model loaders
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.a1 == 'inception' else \
            (256, 128)

    dataset, num_classes, train_loader, val_loader, test_loader, query_loader, multiquery_loader,  gallery_loader = get_data(args.dataset,
                                                                                                         args.split,
                                                                                                         args.data_dir,
                                                                                                         args.height,
                                                                                                         args.width,
                                                                                                         args.batch_size,
                                                                                                         args.workers,
                                                                                                         args.combine_trainval,
                                                                                                         args.loss_mode,
                                                                                                         args.instances_num)


    # Create CNN model, generate 128 dimenional vector through 2 layer fully-connected network
    cnnmodel =  models.create(args.a1, num_features=args.features, dropout=args.dropout)

    # Create the score computation model
    classifiermodel = models.create(args.a2, input_num=args.features)

    # Create the crf_mean_field model
    crfmodel = models.create(args.a3, layer_num=args.layernum)

    # Module cude accelaration
    cnnmodel = nn.DataParallel(cnnmodel).cuda()
    classifiermodel = classifiermodel.cuda()
    crfmodel = crfmodel.cuda()

    # Criterion1 Identiciation loss
    criterion_oim = MULOIMLoss(args.features, num_classes, scalar=args.oim_scalar, momentum= args.oim_momentum)

    # Criterion2 Verification loss
    criterion_veri = PairLoss(args.sampling_rate)

    ## Criterion accerlation cuda
    criterion_oim.cuda()
    criterion_veri.cuda()

    # Optimizer

    base_param_ids = set(map(id, cnnmodel.module.base.parameters()))
    new_params = [p for p in cnnmodel.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': cnnmodel.module.base.parameters(), 'lr_mult': 1},
        {'params': new_params, 'lr_mult': 1},
        {'params': classifiermodel.parameters(), 'lr_mult': 1},
        {'params': crfmodel.parameters(), 'lr_mult': 1}]

    # Optimizer
    optimizer = torch.optim.SGD(param_groups, lr=args.cnnlr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Schedule Learning rate
    def adjust_lr(epoch):
        # step_size = 60 if args.arch == 'inception' else 40
        lr = args.cnnlr * (0.1 ** (epoch //20))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)


    # Trainer
    trainer = MULJOINT_MAN_Trainer(cnnmodel, classifiermodel, crfmodel, criterion_veri, criterion_oim, args.instances_num)
    start_epoch = best_top1 = 0

    # Evaluation
    evaluator = MsEvaluator(cnnmodel, classifiermodel, crfmodel)
    if args.evaluate == 1:
        checkpoint = load_checkpoint(osp.join('../crf_affinity8_models/model101', 'cnncheckpoint.pth.tar'))
        cnnmodel.load_state_dict(checkpoint['state_dict'])

        checkpoint = load_checkpoint(osp.join('../crf_affinity8_models/model101', 'crfcheckpoint.pth.tar'))
        crfmodel.load_state_dict(checkpoint['state_dict'])

        checkpoint = load_checkpoint(osp.join('../crf_affinity8_models/model101', 'classifiercheckpoint.pth.tar'))
        classifiermodel.load_state_dict(checkpoint['state_dict'])

        top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        print(top1)


    else:

        for epoch in range(start_epoch, args.epochs):
            adjust_lr(epoch)
            trainer.train(epoch, train_loader, optimizer)

            if epoch % 6 == 0:
                #print(top1)

                #top1 = top1[0]
                #is_best = top1 > best_top1
                #best_top1 = max(top1, best_top1)
                is_best = True
                save_checkpoint({
                    'state_dict': cnnmodel.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'cnncheckpoint.pth.tar'))

                save_checkpoint({
                    'state_dict': classifiermodel.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'classifiercheckpoint.pth.tar'))


                save_checkpoint({
                    'state_dict': crfmodel.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'crfcheckpoint.pth.tar'))


                top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
                print(top1)

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="script")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    parser.add_argument('--combine-trainval',  default=True)
    # model
    parser.add_argument('--a1', '--arch_1', type=str, default='resfpnnet101',
                        choices=models.names())

    parser.add_argument('--a2', '--arch_2', type=str, default='multiclassifier2',
                        choices=models.names())

    parser.add_argument('--a3', '--arch_3', type=str, default='crf_mf_3_3')

    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--layernum', type=int, default=2)
    parser.add_argument('--evaluate', type=int, default=0)
    # loss
    parser.add_argument('--oim-scalar', type=float, default=30,
                        help='reciprocal of the temperature in OIM loss')
    parser.add_argument('--oim-momentum', type=float, default=0.5,
                        help='momentum for updating the LUT in OIM loss')
    parser.add_argument('--loss-mode', type=str, default='crfloss')
    parser.add_argument('--sampling-rate', type=int, default=5)
    parser.add_argument('--instances_num', type=int, default=4)
    # optimizer
    parser.add_argument('--cnnlr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../datasets'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    main(parser.parse_args())
