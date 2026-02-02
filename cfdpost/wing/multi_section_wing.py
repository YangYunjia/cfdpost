import numpy as np
import math
import copy
from typing import List, Union, Optional
from functools import reduce

from matplotlib.axes import Axes
from scipy.interpolate import CubicSpline

from .basic import BasicParaWing, points2line
from cfdpost.modeling import cst_foil, rotate, dist_clustcos
from cfdpost.utils import DEGREE


class MultiSecWing(BasicParaWing):
    '''
    This is for CRM-liked wings with varying parameters along spanwise
    
    '''

    _format_geometry_indexs = {
        'full': {
            'AoA': 1,
            'Mach': 1,
            'SA'        : 1,   # swept angle (leading edge, deg)
            'DA'        : 1,   # dihedral angle (deg), determine the tip zLE, CubicSpline then applied to calculate 
                                        # outer section dihedral, based on slope at kink (skw, skw1 = 4~6)
            'DA_kink'   : 1,   # dihedral angle at kink, linear inner section (skw, skw1 = 4~6)
            'AR'        : 1,   # aspect ratio
            'TR'        : 1,   # taper ratio (tape)
            'kink'      : 1,   # kink location
            'rootadj'   : 1,   # root kink adjustment (skw, skw1 = 0.5 ~ 1.1 but code was wrong)
            
            'tcroot'    : 1,
            'rtcs'      : 3,   # [kink, kink+tip/2, tip]
            'cambers'   : 3,   # [kink / 2 kink, kink to max, kink+tip/2 (max), tip to max]
            'twists'    : 4,   # tip twist angle (deg) [kink / 2, kink, kink+tip/2, tip]
            'cst_u': 10, 
            'cst_l': 10
        },

        'multi7': {
            'AoA': 1,
            'Mach': 1,
            'SA'        : 1,   # swept angle (leading edge, deg)
            'DA'        : 1,   # dihedral angle (deg), determine the tip zLE, CubicSpline then applied to calculate 
                                        # outer section dihedral, based on slope at kink (skw, skw1 = 4~6)
            'DA_kink'   : 1,   # dihedral angle at kink, linear inner section (skw, skw1 = 4~6)
            'AR'        : 1,   # aspect ratio
            'TR'        : 1,   # taper ratio (tape)
            'kink'      : 1,   # kink location
            'rootadj'   : 1,   # root kink adjustment (skw, skw1 = 0.5 ~ 1.1 but code was wrong)
            
            'control_points': 7,
            'twists'    : 6,   # tip twist angle (deg) [kink / 2, kink, kink+tip/2, tip]
            'cst_u': (10, 7), 
            'cst_l': (10, 7),
        },

        'multi3': {
            'AoA': 1,
            'Mach': 1,
            'SA'        : 1,   # swept angle (leading edge, deg)
            'DA'        : 1,   # dihedral angle (deg), determine the tip zLE, CubicSpline then applied to calculate 
                                        # outer section dihedral, based on slope at kink (skw, skw1 = 4~6)
            'DA_kink'   : 1,   # dihedral angle at kink, linear inner section (skw, skw1 = 4~6)
            'AR'        : 1,   # aspect ratio
            'TR'        : 1,   # taper ratio (tape)
            'kink'      : 1,   # kink location
            'rootadj'   : 1,   # root kink adjustment (skw, skw1 = 0.5 ~ 1.1 but code was wrong)
            
            'twists'    : 4,   # tip twist angle (deg) [kink / 2, kink, kink+tip/2, tip]
            'cst_u': (10, 3), 
            'cst_l': (10, 3),
        },

    }
    
    def __init__(self, paras = None, aoa = None, iscentric = False, normal_factors=(1, 150, 300)):
        super().__init__(paras, aoa, iscentric, normal_factors)
    
    @classmethod
    def _calculate_ref_area(cls, g: dict):
        
        # based on trap. part defination
        if 'ref_area' not in g.keys():
            g['ref_area'] = 0.125 * g['AR'] * (1 + g['TR'])**2
        if 'half_span' not in g.keys():
            g['half_span'] = 0.25 * g['AR'] * (1 + g['TR'])

    @classmethod
    def _reconstruct_surface_grids(cls, g: dict, nx: int, nzs: List[int], tail: float=0.008, 
                                   zaxis=0., lower_cst_constraints: bool = False, twists0: float = 0.) -> np.ndarray:
        '''
        This function reconstructs reference mesh (grid points) of wings for ML models

        It differs from CFD mesh in the following
        - spanwise points are evenly distributed (rather than linear + power growth)
        - airfoil points are cluscos (rather than adapted from CRM-3M mesh)
        - There's no mesh points on the tailing edge

        The others are the same of `~/mount/baselinewings/f6-1/gen-mesh.py`

        At 30.1.2026, we compare the code here with crm_ml_twist/mesh.py and make them are the same

        param of `g`:
        ===

        - for WebWing

            `SA`, `AR`, `TR`, `kink`, `rootadj`, `tmaxs`(7), `DAs`(7), `twists`(7), `cst_u`(7x10), `cst_l`(7x10)

        - for ML opt

            `half_span`, `SA`, `kink`, `chords`(3), `DAs`(7), `twists`(6), `root_twist`, `cst_u`(70), `cst_l`(70 / 63)

        '''

        print('!!!!! building ML mesh')
        
        # get distributed values
        cabin_ratio = 0.1
        half_span = g['half_span'] 
        cabin = half_span * cabin_ratio
        
        # get spanwise eta distribution (evenly distributed)
        yLEs = np.linspace(cabin, half_span, nzs[0])

        # leading edge parameters
        tip = half_span + 1e-9

        if 'control_points' not in g.keys():
            kink = half_span * g['kink']
            if 'DAs' in g.keys():
                _control_points = list(np.concatenate((np.linspace(cabin, kink, 3), np.linspace(kink, half_span, 5)[1:]), axis=0))
                _control_points_mid = _control_points
            else:
                mid_outer = 0.5 * (kink + half_span)
                mid_inner = 0.5 * kink
                _control_points = [0, kink, tip]
                _control_points_mid = [0, mid_inner, kink, mid_outer, tip]
        else:
            kink = half_span * g['control_points'][2]
            _control_points = [half_span * eta for eta in g['control_points']]
            _control_points_mid = _control_points
        
        ### ****
        ### sweep angle
        xLEs = np.tan(g['SA']/DEGREE) * yLEs     # sweep direction
        xLE_kink = np.tan(g['SA']/DEGREE) * kink

        ### ****
        ### dihedral angle
        if 'DA_kink' in g.keys() and 'DA' in g.keys():

            i_kink = math.floor((g['kink']-cabin_ratio)/(1-cabin_ratio) * nzs[0])
            yLEs0 = yLEs[:i_kink+1]
            yLEs1 = yLEs[i_kink+1:]

            zLEs0 = np.tan(g['DA_kink']/DEGREE) * yLEs0  # inner section, linear distribution
            zLE_kink = np.tan(g['DA_kink']/DEGREE) * kink
            zLE_tip = np.tan(g['DA']/DEGREE) * half_span # zLE at kink
            zLE_cs = CubicSpline([kink, tip], [zLE_kink, zLE_tip], bc_type=((1, np.tan(g['DA_kink']/DEGREE)), 'not-a-knot'), extrapolate=False)
            zLEs1 = zLE_cs(yLEs1)
            zLEs  = np.concatenate((zLEs0, zLEs1), axis=0)

        elif 'DAs' in g.keys():
            zLE_cs = CubicSpline([0] + _control_points, [0] + [reduce(lambda x, y: x + y, g['DAs'][:i+1]) for i in range(len(g['DAs']))], extrapolate=False)
            zLEs = zLE_cs(yLEs)

        else:
            raise AttributeError()

        ### ****
        ### chord
        if 'chords' in g.keys():
            chord_cabin, chord_kink, chord_tip = g["chords"]
        else:
            TR = g['TR']
            chord_tip   = TR
            chord_kink  = TR * g['kink'] + 1 * (1 - g['kink'])
            root_adj_l  = g['rootadj'] * (xLE_kink + chord_kink - 1)
            chord_cabin = TR * cabin_ratio + 1 * (1 - cabin_ratio) + root_adj_l * (g['kink'] - cabin_ratio) / g['kink']
            chord_root = 1 + root_adj_l 

        chords = np.where(
            yLEs < kink,
            chord_cabin + (chord_kink - chord_cabin) * (yLEs - cabin) / (kink - cabin),
            chord_kink + (chord_tip - chord_kink) * (yLEs - kink) / (half_span - kink)
        )

        g['surface_area'] = (chord_cabin + chord_kink) * (kink - cabin) * 0.5 + (chord_kink + chord_tip) * (half_span - kink) * 0.5
        
        if 'rtcs' in g.keys():
            
            # root baseline airfoil
            xx1, yu1, yl1, _, _ = cst_foil(nn=nx, cst_u=g['cst_u'], cst_l=g['cst_l'], t=g['tcroot'], tail=tail)
            cmb1 = 0.5 * (yu1 + yl1)
            tc1  = yu1 - yl1
            
            # thickness ratio, camber distribution
            trs_cs = CubicSpline([0, kink, mid_outer, tip], [1] + [reduce(lambda x, y: x * y, g['rtcs'][:i+1]) for i in range(3)], extrapolate=False)
            trs = trs_cs(yLEs)
            g['rtctip'] = trs[-1]
            
            cmb_cs = CubicSpline(_control_points_mid, [0, g['cambers'][0]*g['cambers'][1],
                                                                            g['cambers'][1], 1, g['cambers'][2]], extrapolate=False)
            cmbs = cmb_cs(yLEs)

            yus = np.outer(cmb1, cmbs) + 0.5 * np.outer(tc1, trs) # nn * ns
            yls = np.outer(cmb1, cmbs) - 0.5 * np.outer(tc1, trs) # nn * ns
            zzs = np.concatenate((np.flip(yls[1:], axis=0), yus[:-1]), axis=0)
            
        else:
            # reconstruct airfoils with CSTs and store them in zzss
            zzss = []
            cst_u = np.array(g['cst_u']).reshape(len(_control_points), -1)
            cst_l = np.array(g['cst_l']).reshape(len(_control_points), -1)

            if lower_cst_constraints and cst_u.shape[1] == cst_l.shape[1] + 1:
                cst_l = np.concatenate((-cst_u[:, [0]], cst_l), axis=1)

            for i in range(len(_control_points)):
                xx1, yus, yls, _, _ = cst_foil(nn=nx, cst_u=cst_u[i], cst_l=cst_l[i], t=g['tmaxs'][i] if 'tmaxs' in g.keys() else None, tail=tail)
                zzss.append(np.concatenate((np.flip(yls[1:], axis=0), yus), axis=0))
            
            zzs = []
            for yy in yLEs:
                if yy < _control_points[0]:
                    zzs.append(zzss[0] + (yy - _control_points[0]) / (_control_points[1] - _control_points[0]) * (zzss[1] - zzss[0]))
                elif yy - _control_points[-1] > 0:
                    raise RuntimeError()
                else:
                    for i in range(len(_control_points) - 1):
                        if yy >= _control_points[i] and yy <= _control_points[i+1]:
                            zzs.append(zzss[i] + (yy - _control_points[i]) / (_control_points[i+1] - _control_points[i]) * (zzss[i+1] - zzss[i]))
                            break
                    else:
                        raise RuntimeError
            zzs = np.array(zzs).transpose()
            
        # twist
        twists_angles = ([g['root_twist']] if 'root_twist' in g.keys() else []) + list(g['twists'])

        tws_cs = CubicSpline(_control_points_mid, [reduce(lambda x, y: x + y, twists_angles[:i+1]) + twists0 for i in range(len(twists_angles))], extrapolate=True)
        tws = tws_cs(yLEs)
        zLEs += zaxis * chords * np.sin(tws/180*np.pi)
 
        # rotation
        xxs = np.concatenate((np.flip(xx1[1:], axis=0), xx1), axis=0)

        tws_rat = tws / DEGREE
        xxs1  = np.cos(tws_rat[None, :]) * xxs[:, None] +  np.sin(tws_rat[None, :]) * zzs
        zzs1 = -np.sin(tws_rat[None, :]) * xxs[:, None] +  np.cos(tws_rat[None, :]) * zzs

        xxs1 = xxs1 * chords[None, :] + xLEs[None, :]
        zzs1 = zzs1 * chords[None, :] + zLEs[None, :]

        # print(xxs.shape, np.repeat(yLEs[None, :], 2*nx-1, axis=0).shape, zzs.shape)
        block = np.stack((xxs1, zzs1, np.repeat(yLEs[None, :], 2*nx-1, axis=0)), axis=0).transpose(0, 2, 1) # C, Nz, Ni

        return block
    
    def get_all_geometries(self):
        '''
        Docstring for get_all_geometries
        
        :param self: Description

        return
        origeom: 3, 257, 129
        '''

        origeom = self.geom.transpose(2, 0, 1)
        centricgeom = 0.25 * (origeom[..., 1:,1:] + origeom[..., 1:,:-1] + origeom[..., :-1,1:] + origeom[..., :-1,:-1])
        return origeom, centricgeom
    
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