from __future__ import print_function
import os.path as osp
import numpy as np

from reid.utils.serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.name = 'market1501'

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')



    def load(self, num_val=0, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val

        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        num = len(trainval_pids)



        trainval_pids = sorted(trainval_pids)


        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.trainval = _pluck(identities, trainval_pids, relabel = True)

        if self.name =='cuhk03':
            print('here is CUHK')
            self.query = _pluck(identities, self.split['query'])
            self.gallery = _pluck(identities, self.split['gallery'])

        else:
            query = self.meta['queryset']
            self.query = [tuple(item) for item in query]

            multiquery = self.meta['multiqueryset']
            self.multiquery = [tuple(item) for item in multiquery]

            gallery = self.meta['galleryset']
            self.gallery =[tuple(item) for item in gallery]

        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset         | # ids | # images")
            print("  ---------------------------")
            print("  trainval       | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query          | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))

            print("  multiquery     | {:5d} | {:8d}"
                  .format(len(self.split['multiquery']), len(self.multiquery)))

            print("  gallery        | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
