from __future__ import print_function, absolute_import
import os.path as osp

from .dataset import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Market1501(Dataset):

    url = 'https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'

    def __init__(self, root, split_id=0, num_val=0, download=True):
        super(Market1501, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)


    def download(self):

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)


        # Download the raw zip file
        fpath = osp.join(raw_dir, 'Market-1501-v15.09.15.zip')

        if osp.isfile(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))


        # extract the file
        exdir = osp.join(raw_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)


        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(1502)]


        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()

            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignore
                assert 0 <= pid <= 1501 # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))

                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        def register_test(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            ret = []

            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignore
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))

                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
                ret.append((fname, pid, cam))

            return pids, ret

        def register_multiquery(subdir, query_set, pattern=re.compile(r'([-\d]+)_c(\d)')):

            pids = set()
            ret = []
            for (fname, pid, cam) in query_set:
                filename = '{:04d}_c{:d}*.jpg'.format(pid, cam+1)
                fpaths = sorted(glob(osp.join(exdir, subdir, filename)))

                for fpath in fpaths:
                    fname =osp.basename(fpath)
                    pid, cam = map(int, pattern.search(fname).groups())
                    if pid == -1: continue  # junk images are just ignore
                    assert 0 <= pid <= 1501  # pid == 0 means background
                    assert 1 <= cam <= 6
                    cam -= 1
                    pids.add(pid)
                    fname = ('m{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))

                    identities[pid][cam].append(fname)
                    shutil.copy(fpath, osp.join(images_dir, fname))
                    ret.append((fname, pid, cam))

            return pids, ret


        trainval_pids = register('bounding_box_train')
        gallery_pids, gallery_set = register_test('bounding_box_test')
        query_pids, query_set = register_test('query')
        multiquery_pids, multiquery_set = register_multiquery('gt_bbox', query_set)

        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint((gallery_pids))

        # Save meta information into a json file
        meta = {'name': 'Market1501', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities, 'galleryset': gallery_set, 'queryset': query_set, 'multiqueryset': multiquery_set}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training /test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'multiquery': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]

        write_json(splits, osp.join(self.root, 'splits.json'))
