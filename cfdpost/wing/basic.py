

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
def get_xyforce_1d(geom: np.ndarray, profile: np.ndarray):
    '''
    integrate the force on x and y direction

    param:
    ===
    `geom`:    The geometry (x, y), shape: (2, N)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2

    return:
    ===
    np.ndarray: (Fx, Fy)
    '''
    dfp_n  = (0.5 * (profile[1:] + profile[:-1])).reshape((1, -1))
    dfv_t  = np.zeros_like(dfp_n)
    dr     = (geom[1:] - geom[:-1])

    return np.einsum('lj,lpk,jk->p', np.concatenate((dfv_t, -dfp_n), axis=0), _rot_metrix, dr)

def get_force_1d(geom: np.ndarray, profile: np.ndarray, aoa: float):
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
    dfp = get_xyforce_1d(geom, profile)
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
        self.aoa = aoa

        self._init_blocks()
        self.var_list = []
        self.cl = 0.
        self.cl_curve = None

    def _init_blocks(self):
        self.surface_blocks = []
        self.tail_blocks = []
        self.tip_blocks = []

    def read_prt(self, path, mode='surf'):

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
        '''
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
        sectional_1 = self.surface_blocks[0][0, :, 0]
        sectional_2 = self.surface_blocks[-1][-1, :, 0]

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

    def interpolate_section(self, y):
        blk = self.surface_blocks[0]
        sectional = np.zeros((blk.shape[1], blk.shape[3]))

        for i in range(blk.shape[0] - 1):
            # print(blk[i, 0, 0, 0:3])
            if blk[i, 0, 0, 2] < y and blk[i+1, 0, 0, 2] > y:
                sectional = blk[i, :, 0] + (blk[i+1, :, 0] - blk[i, :, 0]) * (y - blk[i, 0, 0, 2]) / (blk[i+1, 0, 0, 2] - blk[i, 0, 0, 2])
                break
        else:
            print('not found')
        return sectional
    

    def lift_distribution(self):
        blk = self.surface_blocks[0]
        y = blk[:, 0, 0, 2]
        cl = np.zeros((blk.shape[0]))
        cd = np.zeros((blk.shape[0]))

        self.cl = 0.

        for i in range(blk.shape[0]-1):
            cd[i], cl[i] = get_force_1d(blk[i, :, 0, 0:2], blk[i, :, 0, 9], self.aoa)
            self.cl += cl[i] * (y[i+1] - y[i]) * 0.5 / self.g['ref_area']
            if i > 0: self.cl += cl[i] * (y[i] - y[i-1]) * 0.5 / self.g['ref_area']

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

