

from cfdpost.cfdresult import cfl3d
import numpy as np
import math
import copy

from matplotlib import pyplot as plt
from matplotlib import colormaps as cm


_rot_metrix = np.array([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])
DEGREE = 180 / math.pi

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

    def __init__(self, geometry: dict = None, aoa: float = None) -> None:

        self.g = {}
        if geometry is not None:
            self.read_geometry(geometry)

        if 'aoa' in self.g.keys(): self.aoa = self.g['aoa']
        elif 'AoA' in self.g.keys(): self.aoa = self.g['AoA']
        else:
            self.aoa = aoa

        self._init_blocks()
        self.var_list = []
        self.cl = np.zeros((2,))
        self.cl_curve = None

    @property
    def leading_edge_index(self):
        return np.argmin(self.surface_blocks[0][0, :, 0])

    @property
    def cf_expanded(self):
        '''
        check whether the cf_normal and cf_tau is calculated and stored in surface_blocks[0][:, :, 17 and 18]
        '''
        return len(self.surface_blocks[0][0, 0]) > 17 and self.surface_blocks[0][:, :, 17].any() != 0.

    def read_geometry(self, geometry: dict):

        must_keys = ["swept_angle", "dihedral_angle", "aspect_ratio", "tapper_ratio", "tip_twist_angle", "tip2root_thickness_ratio"]

        if sum([kk not in geometry.keys() for kk in must_keys]) > 0:
            raise ValueError('geometry dictionary is missing keys')

        self.g = geometry
        if 'ref_area' not in self.g.keys():
            self.g['ref_area'] = 0.125 * self.g['aspect_ratio'] * (1 + self.g['tapper_ratio'])**2
        if 'half_span' not in self.g.keys():
            self.g['half_span'] = (0.5 * self.g['aspect_ratio'] * self.g['ref_area'])**0.5

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

    def _init_blocks(self):
        self.surface_blocks = []
        # self.tail_blocks = []
        # self.tip_blocks = []

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

    def read_formatted_surface(self, geometry: np.ndarray, data: np.ndarray, isnormed: bool = False):
        self._init_blocks()
        new_block = np.zeros((geometry.shape[1], geometry.shape[2], 19))
        new_block[:, :, 0:3]    = geometry.transpose((1, 2, 0))
        if data.shape[0] == 1:
            new_block[:, :, 9]      = data[0]
        elif data.shape[0] == 2:
            new_block[:, :, 9]      = data[0]
            new_block[:, :, 17]     = data[1] / (1, 200)[isnormed] # cftau
        elif data.shape[0] == 3:
            new_block[:, :, 9]      = data[0]
            new_block[:, :, 15]     = data[2] / (1, 250)[isnormed] # cfy
            new_block[:, :, 17]     = data[1] / (1, 200)[isnormed] # cftau
        self.surface_blocks.append(new_block)

    def _get_normal_cf(self):
        '''
        calculate the cf_normal and cf_tangent

        ### self params:
        - `self.surface_block[0]` should be filled with i_var = 0, 2 (x, z) and 14, 16 (cfx, cfz)

        ### return
        - `cf_tg`, `cf_nm` (`np.ndarray` with size n_sec x n_foil)

        '''
        blk = self.surface_blocks[0]

        if self.cf_expanded:
            print('cf_tau and cf_normal is already calculated')
            return

        xz = np.take(blk, [0, 2], axis=2)
        tangens = np.zeros_like(xz)
        tangens[:, 1:-1] = xz[:, 2:] - xz[:, :-2]
        tangens[:, 0]    = xz[:, 1]  - xz[:, 0]
        tangens[:, -1]   = xz[:, -1] - xz[:, -2]
        tangens = tangens / ((np.sum(tangens**2, axis=2, keepdims=True))**0.5 + 1e-10)
        normals = np.zeros_like(tangens)
        normals[:, :, 0] = -tangens[:, :, 1]
        normals[:, :, 1] = tangens[:, :, 0]
        cfxz = np.take(blk, [14, 16], axis=2)
        cftg = np.einsum('ijk,ijk->ij', tangens, cfxz)
        cfnm = np.einsum('ijk,ijk->ij', normals, cfxz)

        self.surface_blocks[0] = np.dstack((blk, cftg, cfnm))
        # print(cftg[:, 0], self.surface_blocks[0][:, 0, -2])
        # return cftg, cfnm

    @property
    def cf_normal(self):
        if not self.cf_expanded:
            self._get_normal_cf()
        return self.surface_blocks[0][:, :, 17]
        
    def get_formatted_surface(self):
        '''
        get formatted surface data for model training

        ### return

        np.ndarray with size n_sec x n_foil x n_variable (7)
        
        variables:
        ```text
        X, Y, Z, CP, CF_Y, CF_TAU, CF_NOR
        0  1  2   3     4       5       6
        ```
        '''
        if not self.cf_expanded:
            self._get_normal_cf()

        blk = self.surface_blocks[0]
        data = np.take(blk, [0, 1, 2, 9, 15, 17, 18], axis=2)
        return data

    def section_surface_distribution(self, y=None, eta=None, norm=False):

        return interpolate_section(self.surface_blocks[0], y=y, eta=eta, norm=norm)
        
    def lift_distribution(self, vis: bool = True):
        blk = self.surface_blocks[0]
        y = blk[:, 0, 2]
        cl = np.zeros((blk.shape[0]))
        cd = np.zeros((blk.shape[0]))

        self.cl = np.zeros((2,))

        if vis:
            cftg = self.cf_normal
        else:
            cftg = np.zeros_like(blk[:, :, 0])

        for i in range(blk.shape[0]-1):
            cd[i], cl[i] = get_force_1d(blk[i, :, 0:2], self.aoa, blk[i, :, 9], cftg[i])
            self.cl += np.array([cl[i], cd[i]]) * (y[i+1] - y[i]) * 0.5 / self.g['ref_area']
            if i > 0: 
                self.cl += np.array([cl[i], cd[i]]) * (y[i] - y[i-1]) * 0.5 / self.g['ref_area']

        self.cl_curve = (y, cl, cd)
        return y, cl, cd
    
    def sectional_lift_distribution(self):

        if self.cl_curve is None:
            self.lift_distribution()
        
        y, cl, cd = copy.deepcopy(self.cl_curve)
        for i in range(len(y)):
            cl[i] /= self.sectional_chord(y[i])
            cd[i] /= self.sectional_chord(y[i])

        return y, cl, cd
    
    def sectional_chord_eta(self, eta: float | np.ndarray):
        return 1 - (1 - self.g['tapper_ratio']) * eta

    def sectional_chord(self, y: float | np.ndarray):
        return self.sectional_chord_eta(y / self.g['half_span'])
    
    def section_lift_coefficient(self, y):

        if self.cl_curve is None:
            self.lift_distribution()
        
        ys, clss, _ = self.cl_curve
        for i in range(len(ys) - 1):
            if ys[i] < y and ys[i+1] > y:
                section_cl = clss[i] + (clss[i+1] - clss[i]) * (y - ys[i]) / (ys[i+1] - ys[i])
                break
        else:
            raise ValueError('y position not on wing surface')
        
        return section_cl

    def sectional_lift_coefficient(self, y):
        
        return self.section_lift_coefficient(y) / self.sectional_chord(y)

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

        return math.atan(math.tan(sa0 / DEGREE) + chord * 4 * (tr - 1) / (ar * (1 + tr))) * DEGREE
    
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

    def plot_2d(self, contour_option, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                    etas : np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
        
        blk = self.surface_blocks[0]
        surfaces = []
        for op in contour_option:
            if op in ['full']:  surfaces.append(blk)
            elif op in ['upper']:   surfaces.append(blk[:, self.leading_edge_index:])
        
        plot_2d_wing(surfaces[0], surfaces[1], contour, vrange, text, reverse_y, etas, write_to_file)


class KinkWing(Wing):

    def read_geometry(self, geometry: dict):
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

    def sectional_chord_eta(self, eta: float | np.ndarray):
        etak = self.g['kink']
        if eta < etak:
            return 1 - (1 - self.g['tapper_ratio_in']) * eta / etak
        else:
            return self.g['tapper_ratio_in'] * (1 - (1 - self.g['tapper_ratio_ou']) * (eta - etak) / (1 - etak))

    def sectional_chord(self, y: float or np.ndarray):
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

    surfaces = [ww.surface_blocks[0][:, ww.leading_edge_index:] for ww in [wg1, wg2]]
    profiles = [ww.surface_blocks[0] for ww in [wg1, wg2]]

    wg1.lift_distribution(vis=True)
    wg2.lift_distribution(vis=True)

    wg1.g['ground_truth_cl'], wg1.g['ground_truth_cd'] = wg1.cl
    wg1.g['reconstruct_cl'],  wg1.g['reconstruct_cd']  = wg2.cl
    wg1.g['error_cl_(%)'],  wg1.g['error_cd_(%)']  = abs(wg2.cl - wg1.cl) / wg1.cl * 100

    plot_2d_wing(surfaces, profiles, contour, vrange, wg1.g, reverse_y, etas, write_to_file)

def plot_2d_wing(surface, profile_surface=None, contour=4, vrange=(None, None), text: dict = {}, reverse_y=1,
                 etas: np.ndarray = np.linspace(0.1, 0.9, 5), write_to_file = None):
    '''
    plot the wing upper surface and several intersections of the wing

    ### param:

    - `surface`:    the
    
    '''
    from matplotlib.gridspec import GridSpec
    if profile_surface is None:
        profile_surface = surface

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 5, height_ratios=[3, 1])
    # print(blk.shape)
    # print(upper_surface.shape)
    ax = fig.add_subplot(gs[0, :])

    if isinstance(surface, np.ndarray):
        # print(np.max(pp), np.min(pp))
        cs = ax.contourf(surface[:, :, 2], -surface[:, :, 0], surface[:, :, contour], 200, cmap='gist_rainbow', vmin=vrange[0], vmax=vrange[1])
        xmax = surface[-1, -1, 2]
        ax.set_xlim(0, 5)
    elif isinstance(surface, list) and len(surface) == 2:
        cs = ax.contourf(surface[0][:, :, 2], -surface[0][:, :, 0], surface[0][:, :, contour], 200, cmap='gist_rainbow', vmin=vrange[0], vmax=vrange[1])
        cs = ax.contourf(-surface[1][:, :, 2], -surface[1][:, :, 0], surface[1][:, :, contour], 200, cmap='gist_rainbow', vmin=vrange[0], vmax=vrange[1])
        xmax = surface[0][-1, -1, 2]
        ax.set_xlim(-5, 5)

    for eta in etas:
        plt.plot([eta*xmax, eta*xmax], [-3, 0], ls='--', c='k')

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
    colors = ['k', 'r', 'b']
    lss = ['-', '--', '-.']
    for i in range(5):
        if isinstance(profile_surface, np.ndarray):
            profile_surface = [profile_surface]
        ax = fig.add_subplot(gs[1, i])   
        for idx in range(len(profile_surface)):
            sec_p = interpolate_section(profile_surface[idx], eta=etas[i])
            ax.plot(sec_p[:, 0], reverse_y * sec_p[:, contour], c=colors[idx], ls=lss[idx])
    plt.tight_layout()
    if write_to_file is None:
        plt.show()
    else:
        plt.savefig(write_to_file)




if __name__ == '__main__':

    wing_param = {"swept_angle": 0.,
                  "dihedral_angle": 0., 
                  "aspect_ratio": 2 * np.pi, 
                  "tapper_ratio": 1., 
                  "tip_twist_angle": 0., 
                  "tip2root_thickness_ratio": 1.}
    
    wg = Wing(geometry=wing_param)

    xx = wg.thin_wing()
    print(xx)
    dalpha = wg.downwash_angle(np.arange(0.1, 0.9, 0.1), xx)
    print(dalpha)