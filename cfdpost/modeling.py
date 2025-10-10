'''
This file copys useful functions from cst_modeling3d


'''
from typing import Tuple, List, Union, Callable

import copy
import numpy as np

from scipy import spatial
from scipy.special import factorial
from scipy.interpolate import interp1d, CubicSpline
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

#* ===========================================
#* Interpolation
#* ===========================================

def interp_from_curve(x0: Union[float, np.ndarray], x: np.ndarray, y: np.ndarray, 
                        extrapolate=False) -> Union[float, np.ndarray]:
    '''
    Interpolate points from curve represented points [x, y].
    
    Parameters
    ----------
    x0 : Union[float, np.ndarray]
        ndarray/value of x locations to be interpolated.
        
    x, y : ndarray
        coordinates of the curve.

    Returns
    ----------
    y0 : Union[float, np.ndarray]
        interpolated coordinates

    Examples
    ---------
    >>> y0 = interp_from_curve(x0, x, y)
    '''
    if extrapolate:
        f  = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    else:
        f  = interp1d(x, y, kind='cubic')
        
    y0 = f(x0)

    return y0

def find_circle_3p(p1, p2, p3) -> Tuple[float, np.ndarray]:
    '''
    Determine the radius and origin of a circle by 3 points (2D)
    
    Parameters
    -----------
    p1, p2, p3: list or ndarray [2]
        coordinates of points, [x, y]
        
    Returns
    ----------
    R: float
        radius
    XC: ndarray [2]
        circle center

    Examples
    ----------
    >>> R, XC = find_circle_3p(p1, p2, p3)

    '''

    # http://ambrsoft.com/TrigoCalc/Circle3D.htm

    A = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]
    if np.abs(A) <= 1E-20:
        raise Exception('Finding circle: 3 points in one line')
    
    p1s = p1[0]**2 + p1[1]**2
    p2s = p2[0]**2 + p2[1]**2
    p3s = p3[0]**2 + p3[1]**2

    B = p1s*(p3[1]-p2[1]) + p2s*(p1[1]-p3[1]) + p3s*(p2[1]-p1[1])
    C = p1s*(p2[0]-p3[0]) + p2s*(p3[0]-p1[0]) + p3s*(p1[0]-p2[0])
    D = p1s*(p3[0]*p2[1]-p2[0]*p3[1]) + p2s*(p1[0]*p3[1]-p3[0]*p1[1]) + p3s*(p2[0]*p1[1]-p1[0]*p2[1])

    x0 = -B/2/A
    y0 = -C/2/A
    R  = np.sqrt(B**2+C**2-4*A*D)/2/np.abs(A)

    '''
    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x32 = p3[0] - p2[0]
    y32 = p3[1] - p2[1]

    if x21 * y32 - x32 * y21 == 0:
        raise Exception('Finding circle: 3 points in one line')

    xy21 = p2[0]*p2[0] - p1[0]*p1[0] + p2[1]*p2[1] - p1[1]*p1[1]
    xy32 = p3[0]*p3[0] - p2[0]*p2[0] + p3[1]*p3[1] - p2[1]*p2[1]
    
    y0 = (x32 * xy21 - x21 * xy32) / 2 * (y21 * x32 - y32 * x21)
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = np.sqrt(np.power(p1[0]-x0,2) + np.power(p1[1]-y0,2))
    '''

    return R, np.array([x0, y0])

#* ===========================================
#* CST foils
#* ===========================================

def cst_foil(nn: int, cst_u, cst_l, x=None, t=None, tail=0.0, xn1=0.5, xn2=1.0, a0=0.0079, a1=0.96):
    '''
    Constructing upper and lower curves of an airfoil based on CST method

    CST: class shape transformation method (Kulfan, 2008)
    
    Parameters
    -----------
    nn: int
        total amount of points
    cst_u, cst_l: list or ndarray
        CST coefficients of the upper and lower surfaces
    x: ndarray [nn]
        x coordinates in [0,1] (optional)
    t: float
        specified relative maximum thickness (optional)
    tail: float
        relative tail thickness (optional)
    xn1, xn12: float
        CST parameters
        
    Returns
    --------
    x, yu, yl: ndarray
        coordinates
    t0: float
        actual relative maximum thickness
    R0: float
        leading edge radius
    
    Examples
    ---------
    >>> x_, yu, yl, t0, R0 = cst_foil(nn, cst_u, cst_l, x, t, tail)

    '''
    cst_u = np.array(cst_u)
    cst_l = np.array(cst_l)
    x_, yu = cst_curve(nn, cst_u, x=x, xn1=xn1, xn2=xn2, a0=a0, a1=a1)
    x_, yl = cst_curve(nn, cst_l, x=x, xn1=xn1, xn2=xn2, a0=a0, a1=a1)
    
    thick = yu-yl
    it = np.argmax(thick)
    t0 = thick[it]

    # Apply thickness constraint
    if t is not None:
        r  = (t-tail*x_[it])/t0
        t0 = t
        yu = yu * r
        yl = yl * r

    # Add tail
    for i in range(nn):
        yu[i] += 0.5*tail*x_[i]
        yl[i] -= 0.5*tail*x_[i]
        
    # Update t0 after adding tail
    if t is None:
        thick = yu-yl
        it = np.argmax(thick)
        t0 = thick[it]

    # Calculate leading edge radius
    x_RLE = 0.005
    yu_RLE = interp_from_curve(x_RLE, x_, yu)
    yl_RLE = interp_from_curve(x_RLE, x_, yl)
    R0, _ = find_circle_3p([0.0,0.0], [x_RLE,yu_RLE], [x_RLE,yl_RLE])

    return x_, yu, yl, t0, R0

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
        Parameter for distributing points near x=0.
        Smaller a0, more points near x=0.
        
    a1: float
        Parameter for distributing points near x=1.
        Larger a1, more points near x=1.
        
    beta: float
        Parameter for distribution points.

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

def dist_clustcos(nn: int, a0=0.0079, a1=0.96, beta=1.0) -> np.ndarray:
    '''
    Point distribution on x-axis [0, 1]. (More points at both ends)

    Parameters
    ----------
    nn: int
        total amount of points
        
    a0: float
        Parameter for distributing points near x=0.
        Smaller a0, more points near x=0.
        
    a1: float
        Parameter for distributing points near x=1.
        Larger a1, more points near x=1.
        
    beta: float
        Parameter for distribution points.
    
    Examples
    ---------
    >>> xx = dist_clustcos(n, a0, a1, beta)

    '''
    aa = np.power((1-np.cos(a0*np.pi))/2.0, beta)
    dd = np.power((1-np.cos(a1*np.pi))/2.0, beta) - aa
    yt = np.linspace(0.0, 1.0, num=nn)
    a  = np.pi*(a0*(1-yt)+a1*yt)
    xx = (np.power((1-np.cos(a))/2.0,beta)-aa)/dd

    return xx

def cst_curve(nn: int, coef: np.array, x=None, xn1=0.5, xn2=1.0, a0=0.0079, a1=0.96) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generating single curve based on CST method.

    CST: class shape transformation method (Kulfan, 2008)

    Parameters
    ----------
    nn: int
        total amount of points
    coef: ndarray
        CST coefficients
    x: ndarray [nn]
        coordinates of x distribution in [0,1] (optional)
    xn1, xn12: float
        CST parameters
    
    Returns
    --------
    x, y: ndarray
        coordinates
    
    Examples
    ---------
    >>> x, y = cst_curve(nn, coef, x, xn1, xn2)

    '''
    if x is None:
        x = dist_clustcos(nn, a0, a1)
    elif x.shape[0] != nn:
        raise Exception('Specified point distribution has different size %d as input nn %d'%(x.shape[0], nn))
    
    n_cst = coef.shape[0]

    s_psi = np.zeros(nn)
    for i in range(n_cst):
        xk_i_n = factorial(n_cst-1)/factorial(i)/factorial(n_cst-1-i)
        s_psi += coef[i] * xk_i_n * np.power(x, i) * np.power(1 - x, n_cst - 1 - i)

    C_n1n2 = np.power(x, xn1) * np.power(1 - x, xn2)
    y = C_n1n2 * s_psi
    y[0] = 0.0
    y[-1] = 0.0

    return x, y

#* ===========================================
#* Transformation
#* ===========================================

def transform(xu: np.ndarray, xl: np.ndarray, yu: np.ndarray, yl: np.ndarray, 
              scale=1.0, rot=None, x0=None, y0=None, xr=None, yr=None, dx=0.0, dy=0.0, 
              projection=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Apply chord length, twist angle(deg) and leading edge position to a 2D curve.
    
    The transformation is applied in the following order:
    
    1. Translation
    2. Scaling
    3. Rotation

    Parameters
    -------------
    xu, xl, yu, yl : ndarray
        current 2D curve or unit 2D airfoil.
    scale : bool
        scale factor, e.g., chord length.
    rot : {None, float}
        rotate angle (deg), +z direction for x-y plane, e.g., twist angle.
    x0, y0 : float
        coordinates of the scale center.
    xr, yr : float
        coordinates of the rotation center (rotate after translation and scale).
    dx, dy : float
        translation vector, e.g., leading edge location.
    projection : bool
        whether keeps the projection length the same when rotating the section, by default True.

    Returns
    ---------
    xu_new, xl_new, yu_new, yl_new : ndarray
        coordinates of the new 2D curve.
    '''
    #* Translation
    xu_new = dx + xu
    xl_new = dx + xl
    yu_new = dy + yu
    yl_new = dy + yl

    #* Scale center
    if x0 is None:
        x0 = xu_new[0]
    if y0 is None:
        y0 = 0.5*(yu_new[0]+yl_new[0])
    
    #* Scale (keeps the same projection length)
    rr = 1.0
    if projection and not rot is None:
        angle = rot/180.0*np.pi  # rad
        rr = np.cos(angle)

    xu_new = x0 + (xu_new-x0)*scale/rr
    xl_new = x0 + (xl_new-x0)*scale/rr
    yu_new = y0 + (yu_new-y0)*scale/rr
    yl_new = y0 + (yl_new-y0)*scale/rr

    #* Rotation center
    if xr is None:
        xr = x0
    if yr is None:
        yr = y0

    #* Rotation
    if not rot is None:
        xu_new, yu_new, _ = rotate(xu_new, yu_new, np.zeros_like(xu_new), angle=rot, origin=[xr, yr, 0.0], axis='Z')
        xl_new, yl_new, _ = rotate(xl_new, yl_new, np.zeros_like(xu_new), angle=rot, origin=[xr, yr, 0.0], axis='Z')

    return xu_new, xl_new, yu_new, yl_new

def transform_curve(xx: np.ndarray, yy: np.ndarray, dx=0.0, dy=0.0, dz=0.0,
                    scale=1.0, x0=None, y0=None, 
                    rot_z=0.0, rot_x=0.0, rot_y=0.0, rot_axis=0.0,
                    xr=None, yr=None, zr=None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Transform a 2D (unit) curve to a 3D curve by translation, scaling and rotation.
    
    The transformation is applied in the following order:
    
    1. Translation
    2. Scaling
    
        - Scale center: (x0, y0), the first point of the curve by default.
        - Scale factor.
    
    3. Rotation
    
        - Rotate about the z axis by `rot_z` degree.
        - Rotate about the x axis by `rot_x` degree.
        - Rotate about the y axis by `rot_y` degree.
        - Rotate about the main axis of the curve by `rot_axis` degree.
        - Rotate center: (xr, yr, zr), the scale center by default.

    Parameters
    -------------
    xx, yy : ndarray
        a 2D (unit) curve.
        
    dx, dy : float
        translation vector, e.g., leading edge location.
        
    scale : bool
        scale factor.
        
    x0, y0 : float
        the scale center for the 2D curve in the x-y plane.
        
    rot_x, rot_y, rot_z : float
        rotate angle (degree) about the x, y, z axis.
        
    rot_axis : float
        rotate angle (degree) about the main axis of the curve,
        e.g., the chord line of an airfoil.

    xr, yr, zr : float
        the rotation center for the 2D curve

    Returns
    ---------
    x, y, z : ndarray
        coordinates of the 3D curve.
    '''
    #* Translation
    x = dx + xx
    y = dy + yy
    z = dz + np.zeros_like(x)

    #* Scale center
    x0 = x0 if x0 is not None else x[0]
    y0 = y0 if y0 is not None else y[0]
        
    #* Scaling
    x = x0 + (x-x0)*scale
    y = y0 + (y-y0)*scale

    #* Rotation center
    xr = xr if xr is not None else x0
    yr = yr if yr is not None else y0
    zr = zr if zr is not None else dz
    
    #* Rotation
    xv = [1.0]; yv = [0.0]; zv = [0.0]
    
    if abs(rot_z) > 1.0E-12:
        x,  y,  z  = rotate(x,  y,  z,  angle=rot_z, origin=[xr, yr, zr], axis='Z')
        xv, yv, zv = rotate(xv, yv, zv, angle=rot_z, axis='Z')

    if abs(rot_x) > 1.0E-12:
        x,  y,  z  = rotate(x,  y,  z,  angle=rot_x, origin=[xr, yr, zr], axis='X')
        xv, yv, zv = rotate(xv, yv, zv, angle=rot_z, axis='X')

    if abs(rot_y) > 1.0E-12:
        x,  y,  z  = rotate(x,  y,  z,  angle=rot_y, origin=[xr, yr, zr], axis='Y')
        xv, yv, zv = rotate(xv, yv, zv, angle=rot_z, axis='Y')
        
    if abs(rot_axis) > 1.0E-12:
        points = rotate_vector(x, y, z, angle=rot_axis, origin=[xr, yr, zr], axis_vector=[xv[0], yv[0], zv[0]])
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
    return x, y, z

def rotate(x: np.ndarray, y: np.ndarray, z: np.ndarray,
           angle=0.0, origin=[0.0, 0.0, 0.0], axis='X') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Rotate the 3D curve according to origin
    
    Parameters
    ----------
    x, y, z : ndarray
        coordinates of the curve
    angle : float
        rotation angle (deg)
    origin : list of float
        rotation origin
    axis : {'X', 'Y', 'Z'}
        rotation axis (angle is defined by the right-hand rule along this axis)

    Returns
    --------
    x_, y_, z_ : ndarray
        coordinates of the rotated curve
        
    Examples
    --------
    >>> x_, y_, z_ = rotate(x, y, z, angle=0.0, origin=[0.0, 0.0, 0.0], axis='X')
    
    '''
    if axis in 'X':
        axis_vector=[1,0,0]
    if axis in 'Y':
        axis_vector=[0,1,0]
    if axis in 'Z':
        axis_vector=[0,0,1]

    points = rotate_vector(x, y, z, angle=angle, origin=origin, axis_vector=axis_vector)
    
    x_ = points[:,0]
    y_ = points[:,1]
    z_ = points[:,2]

    return x_, y_, z_

def rotate_vector(x, y, z, angle=0, origin=[0, 0, 0], axis_vector=[0,0,1]) -> np.ndarray:
    '''
    Rotate 3D points (vectors) by axis-angle representation.

    Parameters
    ----------
    x, y, z : float or ndarray [:]
        coordinates of the points.
        
    angle : float
        rotation angle (deg) about the axis (right-hand rule).
        
    origin : ndarray [3]
        origin of the rotation axis.
        
    axis_vector : ndarray [3]
        indicating the direction of an axis of rotation.
        The input `axis_vector` will be normalized to a unit vector `e`.
        The rotation vector, or Euler vector, is `angle*e`.

    Returns
    --------
    points : ndarray [3] or [:,3]
        coordinates of the rotated points
        
    Examples
    --------
    >>> points = rotate_vector(x, y, z, angle=0, origin=[0, 0, 0], axis_vector=[0,0,1])
    
    References
    ----------
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
    
    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    
    https://en.wikipedia.org/wiki/Rotation_matrix
    
    '''
    origin = np.array(origin)
    vector = np.transpose(np.array([x, y, z]))  # [3] or [:,3]
    vector = vector - origin
    
    rotation_vector = np.array(axis_vector)/np.linalg.norm(axis_vector)

    rot = Rotation.from_rotvec(angle*rotation_vector/180.0*np.pi)
    
    # In terms of rotation matrices, this application is the same as rot.as_matrix().dot(vector).
    points = rot.apply(vector) + origin
    
    return points

def rotation_3d(pp: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float):
    '''
    The rotation_3d is derived from Chenyu Wu. 2022. 11. 5

    ### Description
    This function rotate a set of points based on the origin and the axis given by the inputs

    ### Inputs
    `pp`: The point set that is going to be rotated. `pp.shape = (n_points, 3)`

    `origin`: The numpy array that defines the origin of the rotation axis. The shape must be `(3,0)`

    `axis`: The direction of the rotation axis. This axis does not need to be normalized. The shape must be `(3,0)`

    `angle`: The rotation angle in degree

    ### Outputs
    `xnew, ynew, znew`: The rotated points.
    '''
    # Translate the points to a coordinate system that has the origin defined by the input
    # The points have to be translated back to the original frame before return.
    
    nn = pp.shape[0]
    for i in range(nn):
        pp[i, :] = pp[i, :] - origin
    xnew, ynew, znew = np.zeros(nn), np.zeros(nn), np.zeros(nn)
    
    norm = np.sqrt(axis @ axis)
    if norm < 1e-8:
        raise Exception("The length of the axis is too short!")
    e3 = axis / norm

    angle_rad = np.pi * angle / 180.0

    for i in range(nn):
        vec = pp[i, :].copy()

        # compute the parallel component
        vec_p = (vec @ e3) * e3
        # compute the normal component
        vec_n = vec - vec_p

        # define the local coordinate system
        e1 = vec_n
        e2 = np.cross(e3, e1)

        # rotate
        vec_n_rot = e1 * np.cos(angle_rad) + e2 * np.sin(angle_rad)

        # assemble the vector
        vec_new = vec_n_rot + vec_p
        xnew[i], ynew[i], znew[i] = vec_new[0], vec_new[1], vec_new[2]

    # transform back to the original frame
    xnew, ynew, znew = xnew + origin[0], ynew + origin[1], znew + origin[2]

    pp_new = np.hstack((xnew.reshape(-1,1), ynew.reshape(-1,1), znew.reshape(-1,1)))
    # print(pp_new.shape)
    return pp_new