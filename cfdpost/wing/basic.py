

from cfdpost.cfdresult import cfl3d
import numpy as np
import math
import copy

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from cst_modeling.section import cst_foil
from cst_modeling.basic import rotate

from typing import List, Union
from cfdpost.utils import DEGREE, get_force_1d, get_moment_1d, get_force_2d, get_moment_2d, get_cellinfo_1d

#* auxilary functions
def _swept_angle(chord: float, sa0: float, ar: float, tr: float) -> float:
    return math.atan(math.tan(sa0 / DEGREE) + chord * 4 * (tr - 1) / (ar * (1 + tr))) * DEGREE

def reconstruct_surface_frame(nx: int, cst_us: List[np.ndarray], cst_ls: List[np.ndarray], ts: List[float], 
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
    for idx, (cst_u, cst_l, t_) in enumerate(zip(cst_us, cst_ls, ts)):
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

class BasicWing():
    
    def __init__(self, paras: dict = None, aoa: float = None, iscentric: bool = False, normal_factors=(1, 150, 300)):
        
        self.g = {}                 # geometric parameters dictionary
                                    # for calculating lift distribution, `half_span` and `ref_area` are required
                                    # which can be calulated either from input planform parameters or input geom-
                                    # tric data with `calc_planform`
        self._init_blocks()         # wing data (only involve single surface blocks now)
        self.cl = np.zeros((3,))    # CL, CD, CM
        self.cl_curve = None        # lift distributions
        
        if isinstance(paras, dict): 
            self.g = paras
        
            if 'aoa' in paras.keys(): self.aoa = paras['aoa']
            elif 'AoA' in paras.keys(): self.aoa = paras['AoA']
        
        if aoa is not None:
            self.aoa = aoa
            
        self.cen = iscentric
        self.normal_factors = normal_factors

    def _init_blocks(self):
        self.surface_blocks = []
        # self.tail_blocks = []
        self.tip_blocks = None    

    def calc_planform(self, force: bool = False):
        '''
        Calculate geometric parameters from grid data
        - halfspan and refarea (projection area)
        - self.leads, self.tails
        - self.chords
        - self.twists
        
        param:
        ===
        `force`: replace the value with calculated from grid data, even if it exist
        
        '''
        blk = self.surface_blocks[0]
        z = blk[:, 0, 2]
        
        if force or 'half_span' not in self.g.keys():
            self.g['half_span'] = max(z)

        if force or 'ref_area' not in self.g.keys():
            x = np.max(blk[:, :, 0], axis=1) - np.min(blk[:, :, 0], axis=1)
            y = blk[:, 0, 1]
            self.g['ref_area'] = np.sum(0.5 * (x[:-1] + x[1:]) * (z[1:] - z[:-1]))

        iLE = self.leading_edge_index
        data = copy.deepcopy(blk[:, :, :3]) # nz, nx, 3

        self.leads = copy.deepcopy(data[:, iLE])
        self.tails = copy.deepcopy(0.5 * (data[:, 0] + data[:, -1])) # nz, 3
        self.chords = np.linalg.norm(self.tails - self.leads, axis=1) # nz`
        self.twists = np.arctan2((self.tails - self.leads)[:, 1], (self.tails - self.leads)[:, 0]) / np.pi * 180

    @property
    def leading_edge_index(self):
        return np.argmin(self.surface_blocks[0][0, :, 0])

    def cf_expanded(self, blk):
        '''
        check whether the cf_normal and cf_tau is calculated and stored in surface_blocks[0][:, :, 17 and 18]
        '''
        return len(blk[0, 0]) > 17 and blk[:, :, 17].any() != 0.

    @property
    def geometry(self):
        '''
        Nz, Nx, Nv
        '''
        
        return self.surface_blocks[0][:, :, :3]
    
    def _read_formatted_surface_geom(self, geometry):
            
        new_block = np.zeros((geometry.shape[1], geometry.shape[2], 19))
        new_block[:, :, 0:3]    = geometry.transpose((1, 2, 0))
            
        return new_block
        
    def _read_formatted_surface_data(self, data, blk, isnormed=False, isxyz=False):
        
        r = -1 if self.cen else None
        
        if data.shape[0] == 1:
            blk[:r, :r, 9]      = data[0]
        elif data.shape[0] == 2:
            if isxyz: raise RuntimeError('read_formatted_surface get 2 channel data, which can not be CFX, CFY, CFZ')
            blk[:r, :r, 9]      = data[0] / (1, self.normal_factors[0])[isnormed]
            blk[:r, :r, 17]     = data[1] / (1, self.normal_factors[1])[isnormed] # cftau
        elif data.shape[0] == 3:
            # cftau, cfz, should be transfer to cfxy
            if isxyz: raise RuntimeError('read_formatted_surface get 3 channel data, which can not be CFX, CFY, CFZ')
            blk[:r, :r, 9]      = data[0] / (1, self.normal_factors[0])[isnormed]
            blk[:r, :r, 16]     = data[2] / (1, self.normal_factors[2])[isnormed] # cfz
            blk[:r, :r, 17]     = data[1] / (1, self.normal_factors[1])[isnormed] # cftau
            self._get_xz_cf(blk)
        # fsw dataset format
        elif data.shape[0] == 4:
            blk[:r, :r, 9]      = data[0]
            
            if isxyz:
                if isnormed: raise RuntimeError('CFX, CFY, CFZ can not be normalized')
                blk[:r, :r, 14:17]     = np.transpose(data[1:4], (1, 2, 0)) # cfx, y, z
                self._get_normal_cf(blk)
            else:  
                blk[:r, :r, 16]     = data[1] / (1, self.normal_factors[2])[isnormed] # cfz
                blk[:r, :r, 17]     = data[2] / (1, self.normal_factors[1])[isnormed] # cftau
                blk[:r, :r, 18]     = data[3] # cfnor

    def read_formatted_surface(self, geometry: np.ndarray = None, data: np.ndarray = None, 
                               isnormed: bool = False, isxyz: bool = False, isinitg: bool = False):
        '''
        read np.ndarray data to the wing

        ### paras:
        - `geometry`:   (N_V, N_Z, N_FOIL) N_V should = 3 (x: streamwise, y, z: spanwise)
        - `data`:   (N_V, N_Z, N_FOIL) 
            - N_V = 1:  only Cp include, will be set to V9
            - N_V = 2:  cp and cftau, will be set to V9 & V17
            - N_V = 3:  cp, cftau, cfz, will be set to V9, V17, V16 (in old version, V9, V15, V17)
        - `isnormed`:   if `True`, cftau & cfz will be divided by 250, 200, respectively
            the non-dim factors was wrong to 200, 250
        - `isxyz`:      if `True`, the input N_V should be 4, and VARS are Cp, Cfx, Cfy, Cfz; they will be automaticly
                        transformed to Cftau and Cfnormal
        - `isinitg`:    if `True`, force to replace half_span and ref_area with grid data, else only fill them if they
                        are not exists in the existing `self.g`
        
        '''
        if geometry is not None:
            if isinstance(geometry, np.ndarray):
                self._init_blocks()
                self.surface_blocks.append(self._read_formatted_surface_geom(geometry))
            elif isinstance(geometry, list):
                self._init_blocks()
                self.surface_blocks.append(self._read_formatted_surface_geom(geometry[0]))
                self.tip_blocks = self._read_formatted_surface_geom(geometry[1])
            
            self.calc_planform(force=isinitg)
            
        if data is not None:
            
            if isinstance(data, np.ndarray):
                self._read_formatted_surface_data(data, self.surface_blocks[0], isnormed, isxyz)
            elif isinstance(data, list):
                self._read_formatted_surface_data(data[0], self.surface_blocks[0], isnormed, isxyz)
                self._read_formatted_surface_data(data[1], self.tip_blocks, isnormed, isxyz)

    def _get_normal_cf(self, blk):
        '''
        calculate the cf_normal and cf_tangent

        ### self params:
        - `self.surface_block[0]` should be filled with i_var = 0, 1 (x, y) and 14, 15 (cfx, cfy)

        Remark: Mar 28, 2024  Correction: spanwise should be z-direction, not y-direction
        - resulting all data (kink 80, kink 160, data 1~3 to be wrong) [marked as 'old']

        ### return
        - `cf_tg`, `cf_nm` (`np.ndarray` with size n_sec x n_foil)

        '''
        if self.cf_expanded(blk):
            print('cf_tau and cf_normal is already calculated')
            return
        
        r = -1 if self.cen else None

        tangens, normals = get_cellinfo_1d(np.take(blk, [0, 1], axis=2), iscentric=self.cen)
        if self.cen:
            tangens = 0.5 * (tangens[1:] + tangens[:-1])    # transfer to cell centre at spanwise direction
            normals = 0.5 * (normals[1:] + normals[:-1])
        
        cf2d = np.take(blk[:r, :r], [14, 15], axis=2)
        cftg = np.sum(cf2d * tangens, axis=-1)
        cfnm = np.sum(cf2d * normals, axis=-1)
        
        # cftg = (blk[:, :, 14]**2 + blk[:, :, 15]**2)**0.5 * np.sign(blk[:, :, 14])
        # cfnm = np.zeros_like(cftg)

        if blk.shape[2] == 19:
            blk[:r, :r, 17:19] = np.dstack((cftg, cfnm))
        else:
            raise RuntimeError('!!!')
        # print(cftg[:, 0], self.surface_blocks[0][:, 0, -2])
        # return cftg, cfnm
        
    def _get_xz_cf(self, blk):
        
        r = -1 if self.cen else None
        
        tangens, normals = get_cellinfo_1d(np.take(blk, [0, 1], axis=2), iscentric=self.cen)
        if self.cen:
            tangens = 0.5 * (tangens[1:] + tangens[:-1])    # transfer to cell centre at spanwise direction
            normals = 0.5 * (normals[1:] + normals[:-1])
        blk[:r, :r, [14, 15]] = blk[:r, :r, [17]] * tangens + blk[:r, :r, [18]] * normals

    @property
    def cf_normal(self):
        if not self.cf_expanded(self.surface_blocks[0]):
            self._get_normal_cf(self.surface_blocks[0])
        return self.surface_blocks[0][:, :, 17]
        
    def get_formatted_surface(self, keep_cen: bool = True):
        '''
        get formatted surface data for model training

        ### return

        np.ndarray with size n_sec x n_foil x n_variable (6)
        
        variables:
        ```text
        X, Y, Z, CP, CF_TAU, CF_Z
        0  1  2   3       4     5
        ```
        remark: the order is changed on Mar 28, 2024 (bet. Cftau & cfz)
        '''
        if not self.cf_expanded(self.surface_blocks[0]):
            self._get_normal_cf(self.surface_blocks[0])
            
        if not keep_cen:  raise NotImplementedError()
        
        blk = self.surface_blocks[0]
        data = blk[..., [0, 1, 2, 9, 17, 16]]
        return data

    def get_normalized_sectional_geom(self):
        '''
        
        normal_geom: nz, 3, nx
        '''

        from cst_modeling.basic import rotate

        data = copy.deepcopy(self.surface_blocks[0][:, :, :3]) # nz, nx, 3
        nz, nx, _ = data.shape
        self.calc_planform()
        alphas = self.g['AoA'] - self.twists
        thicks = self.g['root_thickness'] * (1. - (1. - self.g['tip2root_thickness_ratio']) * np.linspace(0, 1, nz))
        
        elements = (self.leads.transpose(), alphas[np.newaxis, :], self.chords[np.newaxis, :], thicks[np.newaxis, :], 
                        np.tile(self.g['cstu'][0][:, np.newaxis], (1, nz)), np.tile(self.g['cstl'][0][:, np.newaxis], (1, nz)),
                        np.tile(np.array([self.g['Mach']])[:, np.newaxis], (1, nz)))
        #     print([a.shape for a in elements])
        deltaindexs  = np.concatenate(elements, axis=0).transpose()  # Ma
        normal_geom = (data - self.leads[:, np.newaxis, :]) / self.chords[:, np.newaxis, np.newaxis]

        for iz in range(nz):
            normal_geom[iz] = np.stack(rotate(normal_geom[iz, :, 0], normal_geom[iz, :, 1], normal_geom[iz, :, 2], - self.twists[iz], axis='Z')).transpose()

        return deltaindexs, normal_geom.transpose((0, 2, 1))
    
    '''
    def check(self):
        import copy
        data = self.get_formatted_surface()

        ld = self.leading_edge_index
        leads = copy.deepcopy(data[:, ld, :2])
        tails = 0.5 * (data[:, 0, :2] + data[:, -1, :2])
        chords = np.linalg.norm(tails - leads, axis=1)
        alphas = np.arctan2((tails - leads)[:, 1], (tails - leads)[:, 0]) / np.pi * 180

        data1 = np.zeros((*data.shape[:2], 2))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data1[i, j] = _xy_2_cl(data[i, j, :2]-leads[i], alphas[i])
            data1[i] /= chords[i]
            if i == 80:
                plt.plot(data[i, :, 0], data[i, :, 1])
                plt.plot(data1[i, :, 0], data1[i, :, 1])
                plt.savefig('123.png')
    ''' 
    
    def aero_force(self, vis=-1):
        
        if self.cen:
            r = -1
        else:
            raise NotImplementedError()
        
        if vis == -1:
            cf_sel = [14, 15, 16]
        elif vis == 0:
            cf_sel = []
        else:
            cf_sel = [17, 16]
        
        self.cl = np.zeros((3,))
        
        blk = self.surface_blocks[0]
        forces  = get_force_2d(geom=blk[:, :, :3], aoa=self.aoa, cp=blk[:r, :r, 9], cf=blk[:r, :r, cf_sel])
        moments = get_moment_2d(geom=blk[:, :, :3], cp=blk[:r, :r, 9], cf=blk[:r, :r, cf_sel])
        
        if self.tip_blocks is not None:
            blk = self.tip_blocks
            forces_tip = get_force_2d(geom=blk[:, :, :3], aoa=self.aoa, cp=blk[:r, :r, 9], cf=blk[:r, :r, cf_sel])
            moments_tip = get_moment_2d(geom=blk[:, :, :3], cp=blk[:r, :r, 9], cf=blk[:r, :r, cf_sel])
            # print(forces, forces_tip)
            
            forces += forces_tip
            moments += moments_tip
        
        self.cl[1], self.cl[0], _ = forces / self.g['ref_area']
        _, _, self.cl[2]          = moments / self.g['ref_area']
        
        return forces, moments
    
    def section_surface_distribution(self, y=None, eta=None, norm=False):

        return interpolate_section(self.surface_blocks[0], y=y, eta=eta, norm=norm)
        
    def lift_distribution(self, vis: int = 1, keep_cen: bool = True):
        
        if not keep_cen:  raise NotImplementedError()
        
        blk = self.surface_blocks[0]
        r = -1 if self.cen else None
        blk_n = 0.5 * (blk[:-1, 1:] + blk[:-1, :-1]) if self.cen else blk
        
        span_distributions = np.zeros((4, blk_n.shape[0]))
        span_distributions[0] = blk_n[:, 0, 2] if self.cen else blk[:, 0, 2]
        forces = np.zeros((3,))

        if vis == 0:
            cftg = np.zeros_like(blk_n[..., 0])
        # elif vis == -1:
        #     cftg = blk[:, :, 14:16]
        else:
            cftg = blk_n[..., 17]

        for i in range(blk_n.shape[0]):

            cd, cl = get_force_1d(blk[i, :, 0:2], self.aoa, blk_n[i, :, 9], cftg[i, :])
            cmz    = get_moment_1d(blk[i, :, 0:2], blk_n[i, :, 9], cftg[i, :])
            span_distributions[1:, i] = np.array([cl, cd, cmz])
        
        if not self.cen:
            for i in range(blk.shape[0] - 1):
                y, cl, cd, cmz = span_distributions[:, i]
                forces += np.array([cl, cd, cmz]) * (y[i+1] - y[i]) * 0.5
                if i > 0: 
                    forces += np.array([cl, cd, cmz]) * (y[i] - y[i-1]) * 0.5
            self.cl = forces

        self.cl_curve = span_distributions
        return self.cl_curve
    
    def sectional_lift_distribution(self, vis: bool = True):

        self.lift_distribution(vis=vis)

        c = 0.5 * (self.chords[1:] + self.chords[:-1]) if self.cen else self.chords 
        cf_n = copy.deepcopy(self.cl_curve)
        cf_n[1:] /= c[np.newaxis, :]

        return np.concatenate((cf_n[[0]], c[np.newaxis, :], cf_n[1:]))
    
    def sectional_chord_eta(self, eta: Union[float, np.ndarray]):
        return 1 - (1 - self.g['tapper_ratio']) * eta

    def sectional_chord(self, y: Union[float, np.ndarray]):
        return self.sectional_chord_eta(y / self.g['half_span'])
    
    def section_lift_coefficient(self, y):

        if self.cl_curve is None:
            self.lift_distribution()
        
        ys, clss, _, _ = self.cl_curve
        for i in range(len(ys) - 1):
            if ys[i] < y and ys[i+1] > y:
                section_cl = clss[i] + (clss[i+1] - clss[i]) * (y - ys[i]) / (ys[i+1] - ys[i])
                break
        else:
            raise ValueError('y position not on wing surface')
        
        return section_cl

    def sectional_lift_coefficient(self, y):
        
        return self.section_lift_coefficient(y) / self.sectional_chord(y)

    #* =============================
    # below are functions for plot a wing
    def plot_wing(self, contour=4):
        ax = plt.figure().add_subplot(projection='3d')
        
        blk = self.surface_blocks[0]

        def color_map(data, c_map, alpha):
            dmin, dmax = np.nanmin(data), np.nanmax(data)
            _c_map = cm.get_cmap(c_map)
            _colors = _c_map((data - dmin) / (dmax - dmin))
            _colors[:, :, 3] = alpha
            return _colors

        colors = color_map(blk[:, :, contour], 'gist_rainbow', alpha=0.2)

        ax.plot_surface(blk[:, :, 0], blk[:, :, 2], blk[:, :, 1], facecolors=colors, edgecolor='none', rstride=1, cstride=3, shade=True)
        for i in range(0, 60, 10):
            ax.plot(blk[i, :, 0], blk[i, :, 2], -blk[i, :, 9] / 5 + 0.3 + blk[i, 0, 1], c='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.view_init(elev=10, azim=110)
        plt.show()

    def plot_2d(self, contour_option, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                    etas : np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
        
        blk = self.surface_blocks[0]
        surfaces = []
        for op in contour_option:
            if op in ['full']:  surfaces.append(blk)
            elif op in ['upper']:   surfaces.append(blk[:, self.leading_edge_index:])
        
        plot_2d_wing(surfaces[0], surfaces[1], contour, vrange, text, reverse_y, etas, write_to_file)

    def _plot_2d(self, fig: Figure, contour_option: List[str], contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                    etas : np.ndarray = np.linspace(0.1, 0.9, 5)):
        
        blk = self.surface_blocks[0]
        surfaces = []
        for op in contour_option:
            if op in ['full']:  surfaces.append(blk)
            elif op in ['upper']:   surfaces.append(blk[:, self.leading_edge_index:])
        
        _plot_2d_wing(fig, surfaces[0], surfaces[1], contour, vrange, text, reverse_y, etas)

class Wing(BasicWing):
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
    _format_geometry_indexs = ['id', 'AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'tapper_ratio',  
                'tip_twist_angle', 'tip2root_thickness_ratio', 'ref_area']
    _format_geometry_indexs_short = ['AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'tapper_ratio',  
                'tip_twist_angle', 'tip2root_thickness_ratio']
    
    def __init__(self, geometry: Union[list, np.ndarray, dict] = None, aoa: float = None, normal_factors: tuple = (1, 250, 200)) -> None:
        
        super().__init__(paras=geometry, aoa=aoa, init_g=False, normal_factors=normal_factors)
        
        if geometry is not None:
            if isinstance(geometry, dict):   self.read_geometry(geometry)
            elif isinstance(geometry, np.ndarray) or isinstance(geometry, list): self.read_formatted_geometry(geometry)

    def read_geometry(self, geometry: dict, aoa: float = None):

        must_keys = ["swept_angle", "dihedral_angle", "aspect_ratio", "tapper_ratio", "tip_twist_angle", "tip2root_thickness_ratio"]

        if sum([kk not in geometry.keys() for kk in must_keys]) > 0:
            raise ValueError('geometry dictionary is missing keys')

        self.g = geometry
        if 'ref_area' not in self.g.keys():
            self.g['ref_area'] = 0.125 * self.g['aspect_ratio'] * (1 + self.g['tapper_ratio'])**2
        if 'half_span' not in self.g.keys():
            self.g['half_span'] = 0.25 * self.g['aspect_ratio'] * (1 + self.g['tapper_ratio'])

    def read_prt(self, path, mode='surf'):
        '''

        
        ```text
        LE1
        -----
             -------   LE2
                     ---|
        --------        |
        TE1      -------|
                       TE2

        y
        .--> z
        |
        x
        ```

        `self.surface_blocks`:  `list of np.ndarray`
            the dims of `ndarray`: (i_sec (y) x i_foil (xz) x i_variable(17 or 19))
                
                
        variables:
        ```text
        X, Y, Z, U_X, U_Y, U_Z, P, T, M, CP, MUT, DIS, CH, YPLUS, CF_X, CF_Y, CF_Z, (CF_TAU, CF_NOR)
        0  1  2    3    4    5  6  7  8   9   10   11  12     13    14    15    16  (    17      18)
        ```
        '''

        #! this part is not unversial, only for simple wing grid
        self._init_blocks()
        if mode == 'surf':
            xy, qq = cfl3d.readsurf2d(path)
            blocks = [np.concatenate((xy[i], qq[i]), axis=3) for i in range(len(xy))]
        else:
            block_p, block_v = cfl3d.readprt(path)
            blocks = [np.concatenate((block_p[i], block_v[i]), axis=3) for i in range(len(block_p))]

        n_sec = int(len(blocks) / 2)

        _surface_blocks = []

        for i_sec in range(n_sec):
            _surface_blocks.append(blocks[2 * i_sec][int(i_sec > 0):, :, 0])
            # self.tail_blocks.append(blocks[i_sec+1])
            # self.tip_blocks.append(blocks[i_sec+2])
        self.surface_blocks.append(np.concatenate(_surface_blocks))
        # sectional_1 = self.surface_blocks[0][0, :]
        # sectional_2 = self.surface_blocks[-1][-1, :]

        # print(self.surface_blocks[0].shape)

    def read_formatted_geometry(self, geometry: np.ndarray, aoa: float = None, ftype: float = 0):
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
        
        wing_param = {}
        if ftype == 1:
            index_keys = self.__class__._format_geometry_indexs_short
            i_foil_start = 8
        else:
            index_keys = self.__class__._format_geometry_indexs
            i_foil_start = 10
            
        for idx, key in enumerate(index_keys):
            wing_param[key] = float(geometry[idx])

        wing_param['root_thickness'] = float(geometry[i_foil_start])
        wing_param['cstu'] = [np.array(geometry[i_foil_start+1:i_foil_start+11]) for _ in range(2)]
        wing_param['cstl'] = [np.array(geometry[i_foil_start+11:i_foil_start+21]) for _ in range(2)]
        
        self.read_geometry(wing_param)

    def reconstruct_surface_grids(self, nx, nzs, tail=0.004):
        '''
        reconstruct surface grid points (same to generate volume grid points in CFD simulations)

        '''

        troot = self.g['root_thickness']
        xxs, yys = reconstruct_surface_frame(nx, self.g['cstu'], self.g['cstl'], 
                                             [troot, (troot * self.g['tip2root_thickness_ratio'])], self.g)

        # for idx, ny in enumerate(nys):
        idx = 0
        nz = nzs[0]
        blockz = np.tile(np.linspace(0, self.g['half_span'], nz).reshape(1, -1), (2*nx-1, 1))
        blockx = np.outer(xxs[0], np.linspace(1, 0, nz)) + np.outer(xxs[1], np.linspace(0, 1, nz))
        blocky = np.outer(yys[0], np.linspace(1, 0, nz)) + np.outer(yys[1], np.linspace(0, 1, nz))
        
        new_block = np.zeros((nz, 2*nx-1, 19))
        new_block[:, :, :3] = np.stack((blockx, blocky, blockz)).transpose((2, 1, 0))
        self.surface_blocks.append(new_block)

    def reconstruct_strictx_surface_grids(self, cst_us, cst_ls, troot, nx, nzs, tail=0.004):
        '''
        reconstruct surface grid points (same to model training - same x distributions)
        
        '''
        raise NotImplementedError()

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

        return _swept_angle(chord, sa0, ar, tr)
    
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

class KinkWing(Wing):
    
    _format_geometry_indexs = ['id', 'AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'kink', 'tapper_ratio_in', 'tapper_ratio_ou', 
                'tip2root_thickness_ratio', 'tip_twist_in', 'tip_twist_ou', 'ref_area']
    _format_geometry_indexs_short = ['AoA', 'Mach', 'swept_angle', 'dihedral_angle', 'aspect_ratio', 'kink', 'tapper_ratio_in', 'tapper_ratio_ou', 
                'tip2root_thickness_ratio', 'tip_twist_in', 'tip_twist_ou']
    
    def read_geometry(self, geometry: dict, aoa: float = None):
        must_keys = ["swept_angle", "dihedral_angle", "aspect_ratio", "kink", "tapper_ratio_in", "tapper_ratio_ou",
                     "tip_twist_in", "tip_twist_ou", "tip2root_thickness_ratio"]

        if sum([kk not in geometry.keys() for kk in must_keys]) > 0:
            raise ValueError('geometry dictionary is missing keys')

        self.g = geometry
        if 'half_span' not in self.g.keys():
            self.g['half_span'] = 1/4 * self.g['aspect_ratio'] * (self.g['kink'] * (1 + self.g['tapper_ratio_in']) 
                                                               + (1 - self.g['kink']) * self.g['tapper_ratio_in'] * (1 + self.g['tapper_ratio_ou']))
            self.g['ref_area'] = 4 * self.g['half_span']**2 / self.g['aspect_ratio']

            self.g['inner_span'] = self.g['kink'] * self.g['half_span']
            self.g['outer_span'] = (1 - self.g['kink']) * self.g['half_span']

    def sectional_chord_eta(self, eta: Union[float, np.ndarray]):
        etak = self.g['kink']
        if eta < etak:
            return 1 - (1 - self.g['tapper_ratio_in']) * eta / etak
        else:
            return self.g['tapper_ratio_in'] * (1 - (1 - self.g['tapper_ratio_ou']) * (eta - etak) / (1 - etak))

    def sectional_chord(self, y: Union[float, np.ndarray]):
        return self.sectional_chord_eta(y / self.g['half_span'])


def interpolate_section(surface, y=None, eta=None, norm=False):

    # blk = self.surface_blocks[0][:, :, 0]
    sectional = np.zeros((surface.shape[1], surface.shape[2]))

    if y is None:
        if eta is not None:
            y = eta * surface[-1, 0, 2]
        else:
            raise KeyError('at least y or eta should be assigned')

    for i in range(surface.shape[0] - 1):
        # print(blk[i, 0, 0, 0:3])
        if surface[i, 0, 2] <= y and surface[i+1, 0, 2] > y:
            sectional = surface[i, :] + (surface[i+1, :] - surface[i, :]) * (y - surface[i, 0, 2]) / (surface[i+1, 0, 2] - surface[i, 0, 2])
            break
    else:
        print('not found')

    if norm:
        xmin = np.min(sectional[:, 0])
        xmax = np.max(sectional[:, 0])
        sectional[:, 0] = (sectional[:, 0] - xmin) / (xmax - xmin)
        sectional[:, 1] = (sectional[:, 1] - xmin) / (xmax - xmin)

    return sectional

def plot_compare_2d_wing(wg1: Wing, wg2: Wing, contour=4, vrange=(None, None), reverse_y=1, 
                         etas: np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
    
    '''
    - `wg1`:    ground truth
    - `wg2`:    reconstruction
    
    '''

    surfaces = [ww.surface_blocks[0][:, ww.leading_edge_index:] for ww in [wg1, wg2]]
    profiles = [ww.surface_blocks[0] for ww in [wg1, wg2]]

    wg1.lift_distribution(vis=True)
    wg2.lift_distribution(vis=True)

    coefs = ['cl', 'cd', 'cmz']
    for i in range(len(coefs)):
        wg1.g['truth_' + coefs[i]] = wg1.cl[i]
        wg1.g['recons_' + coefs[i]] = wg2.cl[i]
        wg1.g['error(%)_' + coefs[i]] = (abs(wg2.cl - wg1.cl) / wg1.cl * 100)[i]

    plot_2d_wing(surfaces, profiles, contour, vrange, wg1.g, reverse_y, etas, write_to_file)

def plot_2d_wing_surface(ax: Axes, surface, contour=4, vrange=(None, None), text: dict = {},
                 etas: np.ndarray = np.linspace(0.1, 0.9, 5), xrange=(0, 5), yrange=(-3, 0), cmap='gist_rainbow', reverse_value=1):

    # single wing plot -> plot at right half wing
    if isinstance(surface, np.ndarray):
        # print(np.max(pp), np.min(pp))
        cs = ax.contourf(surface[:, :, 2], -surface[:, :, 0], reverse_value * surface[:, :, contour], 200, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        xmax = surface[-1, -1, 2]
        ax.set_xlim(xrange)
    # double wing plot -> for comparison
    elif isinstance(surface, list) and len(surface) == 2:
        cs = ax.contourf(surface[0][:, :, 2], -surface[0][:, :, 0], reverse_value * surface[0][:, :, contour], 200, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        cs = ax.contourf(-surface[1][:, :, 2], -surface[1][:, :, 0], reverse_value * surface[1][:, :, contour], 200, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        xmax = surface[0][-1, -1, 2]
        ax.set_xlim((-xrange[1], xrange[1]))

    # plot slice line to indicate the section positions
    for eta in etas:
        ax.plot([eta*xmax, eta*xmax], [-3, 0], ls='--', c='k')

    ax.set_ylim(yrange)
    ax.set_aspect('equal')

    # plot arrow to indicate freestream
    # ax.arrow(2, 0, 0, -0.4, color='k', head_width=0.05, head_length=0.1)
    # ax.text(2.1, -0.3, 'freestream')
    
    # plot the text if given 
    text_x = 3.5
    text_y = -0.1
    for key in text.keys():
        if isinstance(text[key], float):
            ax.text(text_x, text_y, key + ':     %.4f' % text[key])
            text_y -= 0.1

    return ax, cs

def plot_2d_wing(surface, profile_surface=None, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                 etas: np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
    
    fig = plt.figure(figsize=(14, 10), dpi=100)
    _plot_2d_wing(fig, surface, profile_surface, contour, vrange, text, reverse_y, etas)
    
    plt.tight_layout()
    if write_to_file is None:
        plt.show()
    else:
        plt.savefig(write_to_file)

def _plot_2d_wing(fig: Figure, surface, profile_surface=None, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                 etas: np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
    '''
    plot the wing upper surface and several intersections of the wing

    ### param:

    - `surface`:    the
    
    '''
    if profile_surface is None:
        profile_surface = surface

    gs = GridSpec(2, 5, height_ratios=[3, 1])
    
    # plot the upper surface contour field
    ax = fig.add_subplot(gs[0, :])
    ax, cs = plot_2d_wing_surface(ax, surface, contour, vrange, text, etas)
    cbr = fig.colorbar(cs, fraction=0.01, pad=0.01)

    # plot the section distributions
    colors = ['k', 'r', 'b']
    lss = ['-', '--', '-.']
    for i in range(5):
        if isinstance(profile_surface, np.ndarray):
            profile_surface = [profile_surface]
        ax = fig.add_subplot(gs[1, i])   
        for idx in range(len(profile_surface)):
            sec_p = interpolate_section(profile_surface[idx], eta=etas[i])
            ax.plot(sec_p[:, 0], sec_p[:, contour], c=colors[idx], ls=lss[idx])
        if reverse_y < 0:
            ax.invert_yaxis()


##################
# functions to plot frame view
# all functions below does not rely on Wing / KinkWing classes, for rapid responds
##################

def points2line(p1, p2):
    return tuple([[p1[i], p2[i]] for i in range(len(p1))])
    # return ([p1[0], p2[0]], [p1[1], p2[1]])

def plot_top_view(ax: Axes, sa0: float, ar: float, tr: float):
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

def plot_top_view_kink(ax: Axes, sa0, etak, ar, trin, trout):

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

def plot_top_view_kink1(ax: Axes, sa0, etak, ar, tr, rr):
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

    xxs, yys = reconstruct_surface_frame(nx, [cst_u, cst_u], [cst_l, cst_l], [troot, troot * tcr], g)

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
