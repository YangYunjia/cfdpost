import numpy as np
import math
import copy
from typing import List, Union, Optional

from matplotlib.axes import Axes

from cfdpost.modeling import cst_foil, rotate
from .basic import BasicParaWing, BasicWing, points2line
from cfdpost.utils import DEGREE


class MultiSecWing(BasicWing):
    '''
    This is for CRM-liked wings with varying parameters along spanwise
    
    '''
    def __init__(self, paras = None, aoa = None, iscentric = False, normal_factors=(1, 150, 300)):
        super().__init__(paras, aoa, iscentric, normal_factors)

    #* ploting
    @classmethod
    def plot_top_view(cls, ax: Axes, sa0, etak, ar, tr, rr):
        '''
        new method with tape TR (area is total projection area)
        
        '''
        etar = 0
        pO = [0, 0]
        pA = [0, 1]
        pB = [0, 1 * (1 + rr)]
        half_span = 0.25 * ar * (1 + tr + rr * etak)
        dO1B1 = etar + tr * (1 - etar) + rr * (etak - etar) / etak
        dDE   = etak + tr * (1 - etak)
        pO1 = [half_span*etar, -half_span*etar * np.tan(sa0 / DEGREE)]
        pB1 = [half_span*etar, -half_span*etar * np.tan(sa0 / DEGREE) + dO1B1]
        pD  = [half_span*etak, -half_span*etak * np.tan(sa0 / DEGREE)]
        pE  = [half_span*etak, -half_span*etak * np.tan(sa0 / DEGREE) + dDE]
        pF  = [half_span, -half_span * np.tan(sa0 / DEGREE)]
        pG  = [half_span, -half_span * np.tan(sa0 / DEGREE) + tr]

        ax.plot(*points2line(pO, pF), c='k', ls='-')
        ax.plot(*points2line(pA, pG), c='k', ls='-')
        ax.plot(*points2line(pO, pB), c='k', ls='-')
        ax.plot(*points2line(pD, pE), c='k', ls='--')
        ax.plot(*points2line(pB, pE), c='k', ls='-')
        ax.fill_between([pB1[0], pE[0], pG[0]], [pO1[1], pD[1], pF[1]], [pB1[1], pE[1], pG[1]])

        ax.set_aspect('equal')

        return ax