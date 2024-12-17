

import numpy as np
from scipy.interpolate import interp1d, splev, splrep
# from scipy.optimize import leastsq


class Fitting():
    def __init__(self, X, y, mod='fit', order=4):
        self.mod = mod
        self.order = order
        if mod == 'fit':
            ret = leastsq(Fitting.poly_err, list(np.zeros(self.order +1)), args=(X, y, self.order))
            self.para = ret[0]
        elif mod == 'intp':
            self.para = interp1d(X, y, kind='cubic')
        else:
            raise Exception("ss")
    
    def __call__(self, X):
        if self.mod == 'fit':
            return Fitting.poly_err(self.para, X, 0, self.order)
        elif self.mod == 'intp':
            return self.para(X)
    
    @staticmethod
    def poly_err(p, x, y, order=4):
        value = p[0]
        for i in range(1, order + 1):
            value = value * x + p[i]
        return value - y

DEGREE = 180 / np.pi

def cos(theta):
    return np.cos(theta / DEGREE)

def sin(theta):
    return np.sin(theta / DEGREE)

_rot_metrix = np.array([[[1.0,0], [0,1.0]], [[0,-1.0], [1.0,0]]])

#* function to rotate x-y to aoa

def _aoa_rot(aoa: float) -> np.ndarray:
    '''
    aoa is in size (B, ) with Deg.
    
    '''
    aoa = aoa / DEGREE
    return np.array([np.cos(aoa), -np.sin(aoa)])

def _xy_2_cl(dfp: np.ndarray, aoa: float) -> np.ndarray:
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

def get_dxyforce_1d(geom: np.ndarray, cp: np.ndarray, cf: np.ndarray=None) -> np.ndarray:
    '''
    integrate the force on each surface grid cell

    ### retrun
    np.ndarray (N-1, 2) 
        (dFx, dFy)
    
    '''

    dfp_n  = (0.5 * (cp[1:] + cp[:-1])).reshape((1, -1))
    if cf is None:
        dfv_t  = np.zeros_like(dfp_n)
    else:
        dfv_t = (0.5 * (cf[1:] + cf[:-1])).reshape((1, -1))

    dr     = (geom[1:] - geom[:-1])

    return np.einsum('lj,lpk,jk->jp', np.concatenate((dfv_t, -dfp_n), axis=0), _rot_metrix, dr)


def get_xyforce_1d(geom: np.ndarray, cp: np.ndarray, cf: np.ndarray=None) -> np.ndarray:
    '''
    integrate the force on x and y direction

    param:
    ===
    `geom`:    The geometry (x, y), shape: (N, 2)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
        Cf = tau / 0.5 * rho * U^2

    return:
    ===
    np.ndarray: (Fx, Fy)
    '''

    dr_tail = geom[0] - geom[-1]
    dfp_n_tail = np.array([0., -0.5 * (cp[0] + cp[-1])])
    
    return np.sum(get_dxyforce_1d(geom, cp, cf), axis=0) + np.einsum('l,lpk,k', dfp_n_tail, _rot_metrix, dr_tail)

def get_moment_1d(geom: np.ndarray, cp: np.ndarray, cf: np.ndarray=None, ref_point: np.ndarray=np.array([0.25, 0])) -> np.ndarray:

    dxyforce = get_dxyforce_1d(geom, cp, cf)
    r = 0.5 * (geom[:-1] + geom[1:])
    return np.sum(dxyforce[:, 1] * (r[:, 0] - ref_point[0]) - dxyforce[:, 0] * (r[:, 1] - ref_point[1]))

def get_force_1d(geom: np.ndarray, aoa: float, cp: np.ndarray, cf: np.ndarray=None) -> np.ndarray:
    '''
    integrate the lift and drag

    param:
    ===
    `geom`:    The geometry (x, y), shape: (N, 2)
    
    `profile`: The pressure profile, shape: (N, ); should be non_dimensional pressure profile by freestream condtion

        Cp = (p - p_inf) / 0.5 * rho * U^2
    
    `aoa`:  angle of attack

    return:
    ===
    np.ndarray: (CD, CL)
    '''
    dfp = get_xyforce_1d(geom, cp, cf)
    return _xy_2_cl(dfp, aoa)


def clustcos(i: int, nn: int, a0=0.0079, a1=0.96, beta=1.0) -> float:
    '''
    Point distribution on x-axis [0, 1]. (More points at both ends)
    
    Parameters
    ----------
    i: int
        index of current point (start from 0)
    nn: int
        total amount of points
    a0: float
        parameter for distributing points near x=0
    a1: float
        parameter for distributing points near x=1
    beta: float
        parameter for distribution points 

    Returns
    ---------
    float

    Examples
    ---------
    >>> c = clustcos(i, n, a0, a1, beta)

    '''
    aa = np.power((1-np.cos(a0*np.pi))/2.0, beta)
    dd = np.power((1-np.cos(a1*np.pi))/2.0, beta) - aa
    yt = i/(nn-1.0)
    a  = np.pi*(a0*(1-yt)+a1*yt)
    c  = (np.power((1-np.cos(a))/2.0,beta)-aa)/dd

    return c