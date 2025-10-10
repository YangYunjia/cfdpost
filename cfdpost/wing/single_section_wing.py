import numpy as np
import math
import copy
from typing import List, Union, Optional
from matplotlib.axes import Axes

from cfdpost.modeling import cst_foil, rotate
from .basic import BasicParaWing, points2line
from cfdpost.utils import DEGREE

class Wing(BasicParaWing):
    '''
    must geometry:
        - "swept_angle"   (leading edge)
        - "dihedral_angle"
        - "aspect_ratio"
        - "tapper_ratio"
        - "tip_twist_angle" (deg.)
        - "tip2root_thickness_ratio"
        - "ref_area"
    
    variables:
        ```text
        X, Y, Z, U_X, U_Y, U_Z, P, T, M, CP, MUT, DIS, CH, YPLUS, CF_X, CF_Y, CF_Z, (CF_TAU, CF_NOR)
        0  1  2    3    4    5  6  7  8   9   10   11  12     13    14    15    16  (    17      18)
        ```
    '''
    _format_geometry_indexs = {
        'full': {
            'id': 1, 
            'AoA': 1,
            'Mach': 1,
            'swept_angle': 1, 
            'dihedral_angle': 1, 
            'aspect_ratio': 1, 
            'tapper_ratio': 1,
            'tip_twist_angle': 1,
            'tip2root_thickness_ratio': 1, 
            'ref_area': 1,
            'root_thickness': 1, 
            'cstu': 10, 
            'cstl': 10
        },
        'short': {
            'AoA': 1,
            'Mach': 1,
            'swept_angle': 1, 
            'dihedral_angle': 1, 
            'aspect_ratio': 1, 
            'tapper_ratio': 1,
            'tip_twist_angle': 1,
            'tip2root_thickness_ratio': 1, 
            'root_thickness': 1, 
            'cstu': 10, 
            'cstl': 10
        }
    }
    
    _must_keys = ["swept_angle", "dihedral_angle", "aspect_ratio", "tapper_ratio", "tip_twist_angle", "tip2root_thickness_ratio"]
    
    @classmethod
    def _calculate_ref_area(cls, g: dict):

        if 'ref_area' not in g.keys():
            g['ref_area'] = 0.125 * g['aspect_ratio'] * (1 + g['tapper_ratio'])**2
        if 'half_span' not in g.keys():
            g['half_span'] = 0.25 * g['aspect_ratio'] * (1 + g['tapper_ratio'])

    def read_formatted_geometry(self, geometry: np.ndarray, ftype: float = 0):
        '''
        
        paras:
        ===
        - `type` 
            - `0` :
                'id', 'AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'tapper_ratio',  
                'tip_twist_angle', 'tip2root_thickness_ratio', 'ref_area', 'root_thickness', 'cstu'(10), 'cstl'(10)
            - `1` :
                'AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'tapper_ratio',  
                'tip_twist_angle', 'tip2root_thickness_ratio', 'root_thickness', 'cstu'(10), 'cstl'(10)
        '''
            
        return super().read_formatted_geometry(geometry, index_type=['full', 'short'][ftype])

    @classmethod
    def _reconstruct_surface_frame(cls, nx: int, cst_u: np.ndarray, cst_l: np.ndarray, ts: List[float], 
                              g: dict, tail:float = 0.004):
        '''
        reconstruct the control sectional airfoils

        paras:
        ===
        - `nx`: number of points for each airfoil
        - `cst_us`: `List` of `np.ndarray`, the upper surface CST coefficients from root to tip
        - `cst_ls`: `List` of `np.ndarray`, the lower surface CST coefficients from root to tip
        - `ts`: `List` of `float, the maximum relative thickness from root to tip
        - `g`: `dict`
            - tip_twist_angle
            - tapper_ratio
            - half_span
            - swept_angle
            - dihedral_angle
        '''

        xxs = []
        yys = []
        for idx, t_ in enumerate(ts):
            xx, yu, yl, _, _ = cst_foil(nx, cst_u, cst_l, x=None, t=t_, tail=tail)
            _xx = np.concatenate((xx[::-1], xx[1:]))
            _yy = np.concatenate((yl[::-1], yu[1:]))
            if idx == 1:
                _xx, _yy, _, = rotate(_xx, _yy, np.zeros_like(_xx), angle=g['tip_twist_angle'], origin=[0.0, 0.0, 0.0], axis='Z')
                _xx = g['tapper_ratio'] * _xx + g['half_span'] * np.tan(g['swept_angle']/180*np.pi)
                _yy = g['tapper_ratio'] * _yy + g['half_span'] * np.tan(g['dihedral_angle']/180*np.pi)
            xxs.append(_xx)
            yys.append(_yy)

        return xxs, yys

    @classmethod
    def _reconstruct_surface_grids(cls, g: dict, nx, nzs, tail=0.004):
        '''
        reconstruct surface grid points (same to generate volume grid points in CFD simulations)

        '''

        troot = g['root_thickness']
        xxs, yys = cls._reconstruct_surface_frame(nx, g['cstu'], g['cstl'], 
                                             [troot, (troot * g['tip2root_thickness_ratio'])], g)

        # for idx, ny in enumerate(nys):
        nz = nzs[0]
        blockz = np.tile(np.linspace(0, g['half_span'], nz).reshape(1, -1), (2*nx-1, 1))
        blockx = np.outer(xxs[0], np.linspace(1, 0, nz)) + np.outer(xxs[1], np.linspace(0, 1, nz))
        blocky = np.outer(yys[0], np.linspace(1, 0, nz)) + np.outer(yys[1], np.linspace(0, 1, nz))
        
        return np.stack((blockx, blocky, blockz)).transpose((2, 1, 0))
    
    def reconstruct_surface_grids(self, nx, nzs, tail=0.004):

        new_block = self.__class__._reconstruct_surface_grids(self.g, nx, nzs, tail)
        self.surface_blocks.append([new_block])

    def reconstruct_strictx_surface_grids(self, cst_us, cst_ls, troot, nx, nzs, tail=0.004):
        '''
        reconstruct surface grid points (same to model training - same x distributions)
        
        '''
        raise NotImplementedError()

    @staticmethod
    def _swept_angle(chord: float, sa0: float, ar: float, tr: float) -> float:
        return math.atan(math.tan(sa0 / DEGREE) + chord * 4 * (tr - 1) / (ar * (1 + tr))) * DEGREE

    def swept_angle(self, chord=0.25):
        '''
        The swept angle according to chord section `chord`
        
        return in degree

        '''
        if chord < 0.0  or chord > 1.0:
            raise ValueError('swept baseline should be between 0 and 1, current %.2f' % chord)

        sa0 = self.g['swept_angle']
        if chord == 0:  return sa0

        tr  = self.g['tapper_ratio']
        ar  = self.g['aspect_ratio']

        return self._swept_angle(chord, sa0, ar, tr)

    def sectional_chord_eta(self, eta: Union[float, np.ndarray]) -> float:
        return 1 - (1 - self.g['tapper_ratio']) * eta

    def get_normalized_sectional_geom(self):
        '''
        
        normal_geom: nz, 3, nx
        '''

        data = copy.deepcopy(self.surface_blocks[0][0][:, :, :3]) # nz, nx, 3
        nz, nx, _ = data.shape
        self.calc_planform()
        alphas = self.g['AoA'] - self.twists
        thicks = self.g['root_thickness'] * (1. - (1. - self.g['tip2root_thickness_ratio']) * np.linspace(0, 1, nz))
        
        elements = (self.leads.transpose(), alphas[np.newaxis, :], self.chords[np.newaxis, :], thicks[np.newaxis, :], 
                        np.tile(self.g['cstu'][:, np.newaxis], (1, nz)), np.tile(self.g['cstl'][:, np.newaxis], (1, nz)),
                        np.tile(np.array([self.g['Mach']])[:, np.newaxis], (1, nz)))
        #     print([a.shape for a in elements])
        deltaindexs  = np.concatenate(elements, axis=0).transpose()  # Ma
        normal_geom = (data - self.leads[:, np.newaxis, :]) / self.chords[:, np.newaxis, np.newaxis]

        for iz in range(nz):
            normal_geom[iz] = np.stack(rotate(normal_geom[iz, :, 0], normal_geom[iz, :, 1], normal_geom[iz, :, 2], - self.twists[iz], axis='Z')).transpose()

        return deltaindexs, normal_geom.transpose((0, 2, 1))

    #* =============================
    # below are functions for lifting-line theory

    def thin_wing(self):

        k = 20
        c_alpha = 2 * math.pi

        thetas = (np.arange(k / 2) + 1) / k * np.pi

        mus = (1 - (1 - self.g['tapper_ratio']) * np.cos(thetas)) * c_alpha / (8 * self.g['half_span'])
        a1 = np.array([(nn * mus + np.sin(thetas)) * np.sin(nn * thetas) for nn in range(1, k, 2)])
        b1 = mus * np.sin(thetas)
        xx = np.linalg.solve(a1.transpose(), b1)

        return xx

    def downwash_angle(self, etas, xx):
        
        k = 20

        thetas = np.arccos(np.minimum(etas, 1.0 - 1e-5))
        dacoef = [np.sum(np.arange(1, k, 2) * xx * np.sin(np.arange(1, k, 2) * theta) / np.sin(theta)) / np.pi for theta in thetas]
        return np.array(dacoef)

    #* ploting
    @classmethod
    def plot_top_view(cls, ax: Axes, sa0: float, ar: float, tr: float):
        p1 = [0, 0]
        p2 = [0, 1]
        half_span = 0.25 * ar * (1 + tr)
        p3 = [half_span, -half_span * np.tan(sa0 / DEGREE)]
        p4 = [half_span, -half_span * np.tan(sa0 / DEGREE) + tr]

        ax.plot(*points2line(p1, p2), c='k', ls='-')
        ax.plot(*points2line(p1, p3), c='k', ls='-')
        ax.plot(*points2line(p3, p4), c='k', ls='-')
        ax.plot(*points2line(p2, p4), c='k', ls='-')

        ax.set_aspect('equal')

        return ax

class KinkWing(Wing):
    
    _format_geometry_indexs = {
        'full': {
            'id': 1, 
            'AoA': 1,
            'Mach': 1,
            'swept_angle': 1, 
            'dihedral_angle': 1, 
            'aspect_ratio': 1, 
            'kink': 1,
            'tapper_ratio_in': 1,
            'tapper_ratio_ou': 1,
            'tip2root_thickness_ratio': 1, 
            'tip_twist_angle_in': 1,
            'tip_twist_angle_ou': 1,
            'ref_area': 1,
            'root_thickness': 1, 
            'cstu': 10, 
            'cstl': 10
        },
        'short': {
            'AoA': 1,
            'Mach': 1,
            'swept_angle': 1, 
            'dihedral_angle': 1, 
            'aspect_ratio': 1, 
            'kink': 1,
            'tapper_ratio_in': 1,
            'tapper_ratio_ou': 1,
            'tip2root_thickness_ratio': 1, 
            'tip_twist_angle_in': 1,
            'tip_twist_angle_ou': 1,
            'root_thickness': 1, 
            'cstu': 10, 
            'cstl': 10
        }
    }

    _must_keys = ["swept_angle", "dihedral_angle", "aspect_ratio", "kink", "tapper_ratio_in", "tapper_ratio_ou",
                     "tip_twist_in", "tip_twist_ou", "tip2root_thickness_ratio"]
    
    @classmethod
    def _calculate_ref_area(cls, g: dict):

        if 'half_span' not in g.keys():
            g['half_span'] = 1/4 * g['aspect_ratio'] * (g['kink'] * (1 + g['tapper_ratio_in']) 
                                                               + (1 - g['kink']) * g['tapper_ratio_in'] * (1 + g['tapper_ratio_ou']))
            g['ref_area'] = 4 * g['half_span']**2 / g['aspect_ratio']

            g['inner_span'] = g['kink'] * g['half_span']
            g['outer_span'] = (1 - g['kink']) * g['half_span']

    def sectional_chord_eta(self, eta: Union[float, np.ndarray]):
        etak = self.g['kink']
        if eta < etak:
            return 1 - (1 - self.g['tapper_ratio_in']) * eta / etak
        else:
            return self.g['tapper_ratio_in'] * (1 - (1 - self.g['tapper_ratio_ou']) * (eta - etak) / (1 - etak))

    def sectional_chord(self, y: Union[float, np.ndarray]):
        return self.sectional_chord_eta(y / self.g['half_span'])
    
    #* ploting
    @classmethod
    def plot_top_view(cls, ax: Axes, sa0, etak, ar, trin, trout):

        p1 = [0, 0]
        p2 = [0, 1]
        half_span = 0.25 * ar * (etak * (1 + trin) + (1 - etak) * trin * (1 + trout))
        p3 = [-half_span*etak, half_span*etak * np.tan(sa0 / DEGREE)]
        p4 = [-half_span*etak, half_span*etak * np.tan(sa0 / DEGREE) + trin]
        p5 = [-half_span, half_span * np.tan(sa0 / DEGREE)]
        p6 = [-half_span, half_span * np.tan(sa0 / DEGREE) + trin * trout]

        ax.plot(*points2line(p1, p2), c='k', ls='-')
        ax.plot(*points2line(p1, p5), c='k', ls='-')
        ax.plot(*points2line(p3, p4), c='k', ls='--')
        ax.plot(*points2line(p2, p4), c='k', ls='-')
        ax.plot(*points2line(p5, p6), c='k', ls='-')
        ax.plot(*points2line(p4, p6), c='k', ls='-')

        ax.set_aspect('equal')

        return ax

def plot_frame(ax: Axes, sa0, da0, ar, tr, tw, tcr, troot, cst_u, cst_l) -> Axes:
    
    '''
    ax should be 3D
    '''
    hs = 0.5 * ar * (1 + tr)
    g = {
        'tip_twist_angle': tw,
        'tapper_ratio': tr,
        'half_span': hs,
        'swept_angle': sa0,
        'dihedral_angle': da0
    }
    nx = 51

    xxs, yys = Wing._reconstruct_surface_frame(nx, [cst_u, cst_u], [cst_l, cst_l], [troot, troot * tcr], g)

    # tip and root section airfoil
    ax.plot(xxs[0], [0 for _ in xxs[0]], yys[0] , c='k')
    ax.plot(xxs[1], [hs for _ in xxs[0]], yys[1] , c='k')

    # leading and tailing edges
    for ix in [0, nx-1, -1]:
        ax.plot(*points2line(p1=[xxs[0][ix], 0, yys[0][ix]], p2=[xxs[1][ix], hs, yys[1][ix]]) , c='k')

    # arrow to show freestream direction
    ax.quiver(0, 2, 0, 0.4, 0, 0,
            color='k', arrow_length_ratio=0.2, lw=1,
            pivot='tail', normalize=True)
    ax.text(0, 2, 0, 'freestream')

    return ax