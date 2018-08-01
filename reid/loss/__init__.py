from __future__ import absolute_import

from .oim_loss import oim, OIM, OIMLoss
from .triplet import TripletLoss
from .pairwise import PairwiseLoss
from .crf_loss import CRFLoss
from .pairloss import PairLoss
from .mul_oimloss import MULOIMLoss
from .mul_pairloss import MULPairLoss


__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
]
