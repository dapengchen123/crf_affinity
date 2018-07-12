from __future__ import print_function, absolute_import
import os.path as osp
from reid.datasets import Market1501
from reid.datasets import CUHK03
from reid.datasets import DukeMTMC

from reid.data import transforms as T
from torch.utils.data import DataLoader
from reid.data import Preprocessor
from reid.datasets import create

from reid.data.sampler import RandomPairSampler
from reid.data.sampler import RandomIdentitySampler
from reid.data.sampler import RandomGallerySampler
from reid.data.sampler import RandomMultipleGallerySampler




def get_data(name, split_id, data_dir, height, width, batch_size, workers, combine_trainval, loss_mode='binary_loss', instances_num=4):

    root = osp.join(data_dir, name)
    dataset = create(name, root, split_id=split_id)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)


    train_transformer = T.Compose([T.RandomSizedRectCrop(height, width), T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])


    if loss_mode == 'oim':
        train_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    elif loss_mode == 'binary':

        train_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomPairSampler(train_set), pin_memory=True)

    elif loss_mode == 'triplet':

        train_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(train_set),
            pin_memory=True, drop_last=True)
    elif loss_mode == 'crfloss':

        train_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir,
                         transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomMultipleGallerySampler(train_set, instances_num),
            pin_memory=True, drop_last=True)
    else:
        raise ValueError('NO such loss function')



    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)


    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers = workers,
        shuffle=False, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=64, num_workers=workers,
        shuffle=False, pin_memory=True)

    multiquery_loader = DataLoader(
        Preprocessor(dataset.multiquery,
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=64, num_workers=workers,
        shuffle=False, pin_memory=True
    )

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                    root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        shuffle=False, pin_memory=True)
    
    

    return dataset, num_classes, train_loader, val_loader, test_loader, query_loader, multiquery_loader, gallery_loader
