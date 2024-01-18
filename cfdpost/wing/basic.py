

from cfdpost.cfdresult import cfl3d
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


_rot_metrix = np.array([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])

#* function to rotate x-y to aoa

def _aoa_rot(aoa: float):
    '''
    aoa is in size (B, )
    
    '''
    aoa = aoa * math.pi / 180
    return np.array([math.cos(aoa), -math.sin(aoa)])

def _xy_2_cl(dfp: np.ndarray, aoa: float):
    '''
    transfer fx, fy to CD, CL

    param:
    dfp:    (Fx, Fy), np.ndarray with size (2,)
    aoa:    angle of attack, float

    return:
    ===
    np.ndarray: (CD, CL)
    '''
    # print(dfp.size(), _rot_metrix.size(), _aoa_rot(aoa).size())
    return np.einsum('p,prs,s->r', dfp, _rot_metrix, _aoa_rot(aoa))


#* function to extract pressure force from 1-d pressure profile
def get_xyforce_1d(geom: np.ndarray, cp: np.ndarray, cf: np.ndarray=None):
    '''
    integrate the force on x and y direction

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
        Cf = tau / 0.5 * rho * U^2

    return:
    ===
    np.ndarray: (Fx, Fy)
    '''

    dfp_n  = (0.5 * (cp[1:] + cp[:-1])).reshape((1, -1))
    if cf is None:
        dfv_t  = np.zeros_like(dfp_n)
    else:
        dfv_t = (0.5 * (cf[1:] + cf[:-1])).reshape((1, -1))

    dr     = (geom[1:] - geom[:-1])

    dr_tail = geom[0] - geom[-1]
    dfp_n_tail = np.array([0., -0.5 * (cp[0] + cp[-1])])
    
    return np.einsum('lj,lpk,jk->p', np.concatenate((dfv_t, -dfp_n), axis=0), _rot_metrix, dr) + np.einsum('l,lpk,k', dfp_n_tail, _rot_metrix, dr_tail)

def get_force_1d(geom: np.ndarray, aoa: float, cp: np.ndarray, cf: np.ndarray=None):
    '''
    integrate the lift and drag

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
    
    `aoa`:  angle of attack

    return:
    ===
    np.ndarray: (CD, CL)
    '''
    dfp = get_xyforce_1d(geom, cp, cf)
    return _xy_2_cl(dfp, aoa)



class Wing():

    def __init__(self, geometry: dict, aoa: float = None) -> None:
        '''
        geometry:
            - "swept_angle"
            - "dihedral_angle"
            - "aspect_ratio"
            - "tapper_ratio"
            - "tip_twist_angle"
            - "tip2root_thickness_ratio"
            - "ref_area"
        
        '''

        self.g = geometry
        self.g['maxY'] = (0.5 * self.g['aspect_ratio'] * self.g['ref_area'])**0.5

        if 'aoa' in self.g.keys(): self.aoa = self.g['aoa']
        elif 'AoA' in self.g.keys(): self.aoa = self.g['AoA']
        else:
            self.aoa = aoa

        self._init_blocks()
        self.var_list = []
        self.cl = np.zeros((2,))
        self.cl_curve = None

    def _init_blocks(self):
        self.surface_blocks = []
        self.tail_blocks = []
        self.tip_blocks = []

    def read_prt(self, path, mode='surf'):
        '''
        variables:
        X, Y, Z, U_X, U_Y, U_Z, P, T, M, CP, MUT, DIS, CH, YPLUS, CF_X, CF_Y, CF_Z
        0  1  2    3    4    5  6  7  8   9   10   11  12     13    14    15    16 
        
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
        '''

        #! this part is not unversial, only for simple wing grid
        self._init_blocks()
        if mode == 'surf':
            xy, qq = cfl3d.readsurf2d(path)
            blocks = [np.concatenate((xy[i], qq[i]), axis=3) for i in range(len(xy))]
        else:
            block_p, block_v = cfl3d.readprt(path)
            blocks = [np.concatenate((block_p[i], block_v[i]), axis=3) for i in range(len(block_p))]
        self.surface_blocks.append(blocks[0])
        self.tail_blocks.append(blocks[1])
        self.tip_blocks.append(blocks[2])

        self.leading_edge_index = np.argmin(self.surface_blocks[0][0, :, 0, 0])

        sectional_1 = self.surface_blocks[0][0, :, 0]
        sectional_2 = self.surface_blocks[-1][-1, :, 0]

    def get_normal_cf(self):
        blk = self.surface_blocks[0]
        xz = np.take(blk[:, :, 0], [0, 2], axis=2)
        tangens = np.zeros_like(xz)
        tangens[:, 1:-1] = xz[:, 2:] - xz[:, :-2]
        tangens[:, 0]    = xz[:, 1]  - xz[:, 0]
        tangens[:, -1]   = xz[:, -1] - xz[:, -2]
        tangens = tangens / ((np.sum(tangens**2, axis=2, keepdims=True))**0.5 + 1e-10)
        normals = np.zeros_like(tangens)
        normals[:, :, 0] = -tangens[:, :, 1]
        normals[:, :, 1] = tangens[:, :, 0]
        cfxz = np.take(blk[:, :, 0], [14, 16], axis=2)
        cftg = np.einsum('ijk,ijk->ij', tangens, cfxz)
        cfnm = np.einsum('ijk,ijk->ij', normals, cfxz)
        return cftg, cfnm

    def get_formatted_surface(self):
        blk = self.surface_blocks[0]
        cftg, cfnm = self.get_normal_cf()
        data = np.concatenate((np.take(blk[:, :, 0], [0, 1, 2, 9, 15], axis=2), np.expand_dims(cftg, axis=2), np.expand_dims(cfnm, axis=2)), axis=2)
        # x, y, z, cp, cfy, cftau
        return data

    @staticmethod
    def interpolate_section(surface, y=None, eta=None):

        # blk = self.surface_blocks[0][:, :, 0]
        sectional = np.zeros((surface.shape[1], surface.shape[2]))

        if y is None:
            if eta is not None:
                y = eta * surface[-1, 0, 2]
            else:
                raise KeyError('at least y or eta should be assigned')

        for i in range(surface.shape[0] - 1):
            # print(blk[i, 0, 0, 0:3])
            if surface[i, 0, 2] < y and surface[i+1, 0, 2] > y:
                sectional = surface[i, :] + (surface[i+1, :] - surface[i, :]) * (y - surface[i, 0, 2]) / (surface[i+1, 0, 2] - surface[i, 0, 2])
                break
        else:
            print('not found')
        return sectional
    
    def lift_distribution(self, vis: bool = False):
        blk = self.surface_blocks[0]
        y = blk[:, 0, 0, 2]
        cl = np.zeros((blk.shape[0]))
        cd = np.zeros((blk.shape[0]))

        self.cl = np.zeros((2,))

        if vis:
            cftg, _ = self.get_normal_cf()
        else:
            cftg = np.zeros_like(blk[:, :, 0, 0])

        for i in range(blk.shape[0]-1):
            cd[i], cl[i] = get_force_1d(blk[i, :, 0, 0:2], self.aoa, blk[i, :, 0, 9], cftg[i])
            self.cl += np.array([cl[i], cd[i]]) * (y[i+1] - y[i]) * 0.5 / self.g['ref_area']
            if i > 0: 
                self.cl += np.array([cl[i], cd[i]]) * (y[i] - y[i-1]) * 0.5 / self.g['ref_area']

        self.cl_curve = (y, cl, cd)
        return y, cl, cd
    
    def sectional_lift_distribution(self):

        if self.cl_curve is None:
            self.lift_distribution()
        
        y, cl, cd = self.cl_curve
        for i in range(len(y)):
            cl[i] /= self.sectional_chord(y[i])
            cd[i] /= self.sectional_chord(y[i])

        return y, cl, cd
    
    def sectional_chord(self, y):
        return 1 - (1 - self.g['tapper_ratio']) * y / self.g['maxY']
    
    def sectional_lift_coefficient(self, y):

        if self.cl_curve is None:
            self.lift_distribution()
        
        ys, clss, _ = self.cl_curve
        for i in range(len(ys) - 1):
            if ys[i] < y and ys[i+1] > y:
                sectional_cl = clss[i] + (clss[i+1] - clss[i]) * (y - ys[i]) / (ys[i+1] - ys[i])
                break
        
        return sectional_cl / self.sectional_chord(y)

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

        colors = color_map(blk[:, :, 0, contour], 'gist_rainbow', alpha=0.2)

        ax.plot_surface(blk[:, :, 0, 0], blk[:, :, 0, 2], blk[:, :, 0, 1], facecolors=colors, edgecolor='none', rstride=1, cstride=3, shade=True)
        for i in range(0, 60, 10):
            ax.plot(blk[i, :, 0, 0], blk[i, :, 0, 2], -blk[i, :, 0, 9] / 5 + 0.3 + blk[i, 0, 0, 1], c='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=110)
        plt.show()

    @staticmethod
    def plot_2d_wing(surface, profile_surface=None, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                     write_to_file = None):
        from matplotlib.gridspec import GridSpec
        if profile_surface is None:
            profile_surface = surface

        fig = plt.figure(figsize=(14, 10))

        gs = GridSpec(2, 5, height_ratios=[3, 1])
        # print(blk.shape)
        # print(upper_surface.shape)
        pp = surface[:, :, contour]
        # print(np.max(pp), np.min(pp))
        ax = fig.add_subplot(gs[0, :])
        cs = ax.contourf(surface[:, :, 2], -surface[:, :, 0], pp, 200, cmap='gist_rainbow', vmin=vrange[0], vmax=vrange[1])
        ax.set_xlim(0, 5)
        ax.set_ylim(-3, 0)
        ax.set_aspect('equal')
        cbr = fig.colorbar(cs, fraction=0.01, pad=0.01)
        
        text_x = 3.5
        text_y = -0.1
        for key in text.keys():
            if isinstance(text[key], float):
                ax.text(text_x, text_y, key + ':     %.4f' % text[key])
                text_y -= 0.1
        # plt.show()

        # plt.figure(figsize=(2, 10))
        for i in range(5):
            sec_p = Wing.interpolate_section(profile_surface, eta=0.1+0.2*i)
            ax = fig.add_subplot(gs[1, i])   
            ax.plot(sec_p[:, 0], reverse_y * sec_p[:, contour], c='k')
        plt.tight_layout()
        if write_to_file is None:
            plt.show()
        else:
            plt.savefig(write_to_file)

