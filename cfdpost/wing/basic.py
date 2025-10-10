'''
basic wing definition


'''

from cfdpost.cfdresult import cfl3d
import numpy as np
import math
import copy

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from typing import List, Union, Optional
from cfdpost.modeling import cst_foil, rotate
from cfdpost.utils import DEGREE, get_force_1d, get_moment_1d, get_force_2d, get_moment_2d, get_cellinfo_1d


class BasicWing():
    '''
    This class only deal with surface pressure / friction distributions, and doesn't deal with particule the 
    shape parameters i.e., tapper ratio, aspect ratio and so on. This means every output given by this class
    rely only on the distributed data. (i.e., the chords are calculated from measuring the distance between 
    LE to TE, rather to get from TR)

    The geometry parameters only include:
    - half_span
    - ref_area
    
    angle of attack can be entered in either args aoa (primary), or in dict paras

    '''
    
    def __init__(self, paras: dict = None, aoa: float = None, iscentric: bool = False, normal_factors=(1, 150, 300)):
        
        self.g = {}                 # geometric parameters dictionary
                                    # for calculating lift distribution, `half_span` and `ref_area` are required
                                    # which can be calulated either from input planform parameters or input geom-
                                    # tric data with `calc_planform`

        self._init_blocks()         # wing data (only involve single surface blocks now)
        self.coefficients = np.zeros((3,))    # CL, CD, CMz
        self.span_distrib = None        # lift distributions
        
        if isinstance(paras, dict): 
            self.g = paras
        
            if 'aoa' in paras.keys(): self.aoa = paras['aoa']
            elif 'AoA' in paras.keys(): self.aoa = paras['AoA']
        
        if aoa is not None:
            self.aoa = aoa
        
        # for storage values
        self.store_variables = {}
        self.is_centric = iscentric
        self.normal_factors = normal_factors

    def _init_blocks(self):
        '''
        blocks <= each block <= (geom, data)
                        geom: span, airfoil, (x,y,z) => z = spanwise
                        data: span, airfoil, (Cp, Cfx, Cfy, Cfz, Cfn)
        
        '''
        self.surface_blocks = []
        # self.tail_blocks = []
        self.tip_blocks = None    

    @property
    def more_than_one_block(self):
        return len(self.surface_blocks) > 1

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
        if self.more_than_one_block: raise NotImplementedError

        blk = self.surface_blocks[0][0]
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
        if self.more_than_one_block: raise NotImplementedError
        return np.argmin(self.surface_blocks[0][0][0, :, 0])
    
    def _read_formatted_surface_geom(self, geometry):
        '''
        input:
        ===
        `np.array` => (nz, ni, (x, y, z)) , z=spanwise
        
        '''
            
        return [copy.deepcopy(geometry.transpose((1, 2, 0)))]
        
    def _read_formatted_surface_data(self, data, blk, isnormed=False, isxyz=False):
        '''
        input:
        ===
        `np.array` => (nz, ni, nv) , z=spanwise

        nv =>
        - `nv=1` => `Cp`
        
        '''
                
        if data.shape[0] == 1:
            self.store_variables = {'cp': 0}
        elif data.shape[0] == 2:
            if isxyz: raise RuntimeError('read_formatted_surface get 2 channel data, which can not be CFX, CFY, CFZ')
            self.store_variables = {'cp': 0, 'cft': 1}
        elif data.shape[0] == 3:
            # cftau, cfz, should be transfer to cfxy
            if isxyz: raise RuntimeError('read_formatted_surface get 3 channel data, which can not be CFX, CFY, CFZ')
            self.store_variables = {'cp': 0, 'cft': 1, 'cfz': 2}
        elif data.shape[0] == 4:
            if isxyz:
                if isnormed: raise RuntimeError('CFX, CFY, CFZ can not be normalized')
                self.store_variables = {'cp': 0, 'cfx': 1, 'cfy': 2, 'cfz': 3}
            else:
                # fsw dataset format
                self.store_variables = {'cp': 0, 'cfz': 1, 'cft': 2, 'cfn': 3}
        else:
            raise RuntimeError()
        
        raw_data = copy.deepcopy(data)
        if isnormed:
            v_names = ['cp', 'cft', 'cfz']
            for i in range(len(self.normal_factors)):
                if v_names[i] in self.store_variables.keys():
                    raw_data[self.store_variables[v_names[i]]] /= self.normal_factors[i]
        
        blk.append(raw_data.transpose((1, 2, 0)))

        if 'cfx' not in self.store_variables and 'cft' in self.store_variables:
            self._get_xz_cf(blk)

        if 'cft' not in self.store_variables and 'cfx' in self.store_variables:
            self._get_normal_cf(blk)

    def read_formatted_surface(self, geometry: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None, 
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

        _surface_blocks = np.concatenate(_surface_blocks)
        self.surface_blocks.append([_surface_blocks[:, :, [0, 1, 2]], _surface_blocks[:, :, [9, 14, 15, 16]]])
        self.store_variables = {'cp': 0, 'cfx': 1, 'cfy': 2, 'cfz': 3}
        self.is_centric = False
        # sectional_1 = self.surface_blocks[0][0, :]
        # sectional_2 = self.surface_blocks[-1][-1, :]

        # print(self.surface_blocks[0].shape)

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
        if 'cft' in self.store_variables:
            print('cf_tau and cf_normal is already calculated')
            return
        
        tangens, normals = get_cellinfo_1d(blk[0][:, :, [0,1]], iscentric=self.is_centric)

        if self.is_centric:
            tangens = 0.5 * (tangens[1:] + tangens[:-1])    # transfer to cell center at spanwise direction
            normals = 0.5 * (normals[1:] + normals[:-1])
        
        cf2d = blk[1][:, :, [self.store_variables['cfx'], self.store_variables['cfy']]]
        cftg = np.sum(cf2d * tangens, axis=-1)
        cfnm = np.sum(cf2d * normals, axis=-1)
        
        # cftg = (blk[:, :, 14]**2 + blk[:, :, 15]**2)**0.5 * np.sign(blk[:, :, 14])
        # cfnm = np.zeros_like(cftg)
        self.store_variables['cft'] = blk[1].shape[2]
        self.store_variables['cfn'] = blk[1].shape[2] + 1
        blk[1] = np.concatenate((blk[1], np.dstack((cftg, cfnm))), axis=2)

        # print(cftg[:, 0], self.surface_blocks[0][:, 0, -2])
        # return cftg, cfnm
        
    def _get_xz_cf(self, blk):

        if 'cfx' in self.store_variables:
            print('cf_tau and cf_normal is already calculated')
            return

        tangens, normals = get_cellinfo_1d(blk[0][:, :, [0,1]], iscentric=self.is_centric)
        if self.is_centric:
            tangens = 0.5 * (tangens[1:] + tangens[:-1])    # transfer to cell centre at spanwise direction
            normals = 0.5 * (normals[1:] + normals[:-1])
        
        self.store_variables['cfx'] = blk[1].shape[2]
        self.store_variables['cfy'] = blk[1].shape[2] + 1

        if 'cfn' not in self.store_variables:
            cfn = np.zeros_like(blk[1][:, :, [self.store_variables['cft']]])
        else:
            cfn = blk[1][:, :, [self.store_variables['cfn']]]

        blk[1] = np.concatenate((blk[1], blk[1][:, :, [self.store_variables['cft']]] * tangens + cfn * normals), axis=2)

    def get_formatted_surface(self, keep_cen: bool = True) -> np.ndarray:
        '''
        get formatted surface data for model training

        NOT work for centric

        ### return

        np.ndarray with size n_sec x n_foil x n_variable (6)
        
        variables:
        ```text
        X, Y, Z, CP, CF_TAU, CF_Z
        0  1  2   3       4     5
        ```
        remark: the order is changed on Mar 28, 2024 (bet. Cftau & cfz)
        '''
        if 'cft' not in self.store_variables.keys():
            self._get_normal_cf(self.surface_blocks[0])
        
        if self.is_centric: raise NotImplementedError()

        if not keep_cen:  raise NotImplementedError()
        
        blk = self.surface_blocks[0]
        data = np.concatenate((blk[0], blk[1][..., [self.store_variables['cp'], self.store_variables['cft'], self.store_variables['cfz']]]), axis=2)
        return data

    @property
    def geom(self) -> np.ndarray:
        '''
        return:
        ===
        `np.ndarray` => (nz, ni, 3)
        '''
        if self.more_than_one_block: raise NotImplementedError
        return self.surface_blocks[0][0]
    
    @property
    def cp(self) -> np.ndarray:
        '''
        return:
        ===
        `np.ndarray` => (nz, ni)
        '''
        if self.more_than_one_block: raise NotImplementedError
        return self.surface_blocks[0][1][..., [self.store_variables['cp']]]
    
    @property
    def cf_vector(self) -> np.ndarray:
        '''
        return:
        ===
        `np.ndarray` => (nz, ni, nv)
        
        '''
        if self.more_than_one_block: raise NotImplementedError
        return self.surface_blocks[0][1][..., [self.store_variables['cfx'], self.store_variables['cfy'], self.store_variables['cfz']]]
    
    @property
    def cf_norm(self) -> np.ndarray:
        '''
        return:
        ===
        `np.ndarray` => (nz, ni)
        
        '''
        if self.more_than_one_block: raise NotImplementedError
        return np.linalg.norm(self.cf_vector, axis=2)

    def aero_force(self, vis=-1):
        '''
        integral of aerodynamic forces from geometry and pressure / friction distribution
        
        '''

        def get_blk(blk):
            if vis == -1:
                cf_sel = [self.store_variables['cfx'], self.store_variables['cfy'], self.store_variables['cfz']]
            elif vis == 0:
                cf_sel = []
            else:
                cf_sel = [self.store_variables['cft'], self.store_variables['cfz']]

            if not self.is_centric:
                cen_blk = 0.25 * (blk[1][1:, 1:] + blk[1][1:, :-1] + blk[1][:-1, 1:] + blk[1][:-1, :-1])
            else:
                cen_blk = blk[1]

            geom = blk[0]
            cp = cen_blk[:, :, self.store_variables['cp']]
            cf = cen_blk[:, :, cf_sel]
            return geom, cp, cf

        self.coefficients = np.zeros((3,))
        
        geom, cp, cf = get_blk(self.surface_blocks[0])
        forces  = get_force_2d(geom=geom, aoa=self.aoa, cp=cp, cf=cf)
        moments = get_moment_2d(geom=geom, cp=cp, cf=cf)
        
        if self.tip_blocks is not None:
            geom, cp, cf = get_blk(self.tip_blocks)
            forces_tip = get_force_2d(geom=geom, aoa=self.aoa, cp=cp, cf=cf)
            moments_tip = get_moment_2d(geom=geom, cp=cp, cf=cf)
            # print(forces, forces_tip)
            
            forces += forces_tip
            moments += moments_tip
        
        self.coefficients[1], self.coefficients[0], _ = forces / self.g['ref_area']
        _, _, self.coefficients[2]          = moments / self.g['ref_area']
        
        return forces, moments
    
    def section_surface_distribution(self, y=None, eta=None, norm=False):

        return interpolate_section(self.surface_blocks[0], y=y, eta=eta, norm=norm)
        
    def lift_distribution(self, vis: int = 1, update_force: bool = False):
        
        if self.more_than_one_block: raise NotImplementedError()
        
        blk = self.surface_blocks[0]

        geoms = copy.deepcopy(blk[0])
        cps = copy.deepcopy(blk[1][:, :, self.store_variables['cp']])
        if vis == 0:
            cfs = np.zeros_like(cps)
        else:
            cfs = copy.deepcopy(blk[1][:, :, self.store_variables['cft']])

        if not self.is_centric:
            cps = 0.5 * (cps[:, 1:] + cps[:, :-1])
            cfs = 0.5 * (cfs[:, 1:] + cfs[:, :-1])
        else:
            geoms = 0.5 * (geoms[1:] + geoms[:-1])
        
        span_distributions = np.zeros((4, blk[1].shape[0]))
        span_distributions[0] = geoms[:, 0, 2]

        coefs = get_force_1d(geoms[..., :2], self.aoa * np.ones((geoms.shape[0])), cps, cfs)
        cmzs  = get_moment_1d(geoms[..., :2], cps, cfs)

        span_distributions[1:, :] = np.array([coefs[:, 1], coefs[:, 0], cmzs])
        
        if update_force:
            # there is bug!
            raise NotImplementedError()
            forces = np.zeros((3,))
            y = span_distributions[0]
            for i in range(span_distributions.shape[0] - 1):
                forces += np.array(span_distributions[1:, i]) * (y[i+1] - y[i]) * 0.5
                if i > 0: 
                    forces += np.array(span_distributions[1:, i]) * (y[i] - y[i-1]) * 0.5
            self.coefficients = forces

        self.span_distrib = span_distributions
        return self.span_distrib
    
    def sectional_lift_distribution(self, vis: bool = True):

        self.lift_distribution(vis=vis)

        c = 0.5 * (self.chords[1:] + self.chords[:-1]) if self.is_centric else self.chords 
        cf_n = copy.deepcopy(self.span_distrib)
        cf_n[1:] /= c[np.newaxis, :]

        return np.concatenate((cf_n[[0]], c[np.newaxis, :], cf_n[1:]))
    
    def sectional_chord_eta(self, eta: Union[float, np.ndarray]) -> float:
        raise NotImplementedError

    def sectional_chord(self, y: Union[float, np.ndarray]):
        return self.sectional_chord_eta(y / self.g['half_span'])
    
    def section_lift_coefficient(self, y):

        if self.span_distrib is None:
            self.lift_distribution()
        
        ys, clss, _, _ = self.span_distrib
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

class BasicParaWing(BasicWing):
    '''
    Base class for parameterized wings
    
    '''
    _format_geometry_indexs = {'full': {}}           # include others
    _must_keys = []


    def __init__(self, geometry: Union[list, np.ndarray, dict] = None, aoa: float = None, normal_factors: tuple = (1, 250, 200)) -> None:
        
        super().__init__(paras=None, aoa=aoa, normal_factors=normal_factors)
        
        if geometry is not None:
            if isinstance(geometry, dict):   self.read_geometry(geometry)
            elif isinstance(geometry, np.ndarray) or isinstance(geometry, list): self.read_formatted_geometry(geometry)

    def read_geometry(self, geometry: dict) -> dict:
        '''
        read geometry from parameters
        '''

        if sum([kk not in geometry.keys() for kk in self.__class__._must_keys]) > 0:
            raise ValueError('geometry dictionary is missing keys')
        
        self.__class__._calculate_ref_area(geometry)

        self.g = geometry

    def read_formatted_geometry(self, geometry: np.ndarray, index_type: str = 'full') -> dict:
        '''
        read formatted geometries given the _format_geometry_indexs
        '''
        
        wing_param = {}
        index_keys: dict = self.__class__._format_geometry_indexs[index_type]

        idx = 0
        for k in index_keys.keys():
            if index_keys[k] == 1:
                wing_param[k] = float(geometry[idx])
                idx += 1
            else:
                wing_param[k] = np.array(geometry[idx:idx + index_keys[k]])
                idx += index_keys[k]

        self.read_geometry(wing_param)

    @classmethod
    def _calculate_ref_area(cls, g: dict) -> None:
        '''
        calculate reference area and half span from primary input
        
        '''
        raise NotImplementedError()
    
    def calculate_ref_area(self):
        self.__class__._calculate_ref_area(self.g)



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

def plot_compare_2d_wing(wg1: BasicWing, wg2: BasicWing, contour=4, vrange=(None, None), reverse_y=1, 
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
        wg1.g['truth_' + coefs[i]] = wg1.coefficients[i]
        wg1.g['recons_' + coefs[i]] = wg2.coefficients[i]
        wg1.g['error(%)_' + coefs[i]] = (abs(wg2.coefficients - wg1.coefficients) / wg1.coefficients * 100)[i]

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
