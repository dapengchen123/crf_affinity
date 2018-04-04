from __future__ import absolute_import
from .market1501 import Market1501
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC


__factory = {
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
}

def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)