'''
Extract physical features of airfoils or wing sections.
'''
import copy
import os

import numpy as np
import cfdpost.utils as ut
from scipy.interpolate import interp1d

from typing import Tuple, List


DATA_TYPE = ['DIMENTIONAL']


class PhysicalLine():

    '''
    base class for linear data

    xf_dict:  dictionary of flow features
        the value of each flow features should contain 3 parts:
        1. full name of the feature
        2. index of the feature position (int or List[int])
        3. x position of the feature position (float or List[float])
        remark that 3 is not the position of 2

    initial para:
    ===
    `name`
    
    '''

    xf_dict = {}

    def __init__(self, name='new_physical'):

        self.name = name
        self.xf_dict = copy.deepcopy(self.__class__.xf_dict)
        # self._basic_var_info = basic_var_info

    def setdata(self, x, y, Cp, Tw, dudy, info={'x': 'DIMENTONAL', 'Cp': 'P', 'dudy':'Tau_x'}, **kwargs):
        '''
        Set the data of this foil or section.

        Data:   ndarray, start from lower surface trailing edge
        '''
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.Cp = copy.deepcopy(Cp)
        self.Tw = copy.deepcopy(Tw)
        self.dudy = copy.deepcopy(dudy)
        self.info = info

    @property
    def n_point(self):
        '''
        Number of points in this section
        '''
        return self.x.shape[0]

    def _get_i0_i1(self, ii: int) -> Tuple[int, int]:
        '''
        return the index of the line interpolating for the other variables
        for a wall section, all the input data could be used to interpolate, so return none
        for a surface whose `x` is not increasing, a function should be implenment to 
        guarantee the data used to interpolate is increasing
        '''
        raise NotImplementedError

    def getValue(self, feature: str, key: str = 'i') -> float or List[float]:
        '''
        Get value of given feature.

        ### Inputs:
        ```text
        feature:    key of feature dictionary
        key:        'i', 'X', 'Cp', 'Mw', 'Tw', 'Hi', 'Hc', 'dudy'
        ```
        '''

        if not (feature in self.xf_dict.keys()):
            print('  Warning: feature [%s] not valid'%(feature))
            return 0.0

        # find the feature position
        aa = self.xf_dict[feature]

        if key in ['i']:
            return aa[1]

        if key in ['X']:
            return aa[2]

        # print out other varibles at `feature` position

        if not key in self.__dict__:
            raise KeyError('  Warning: key [%s] not valid'%(key))
        
        ii = aa[1]
        xx = aa[2]

        # single position, aa[1] is int indicating index
        if not isinstance(ii, list):
            i0, i1 = self._get_i0_i1(ii)
            return self.getValueX(xx, key=key, i0=i0, i1=i1)
        
        # multiple position, aa[1] is list of int indicating index
        else:
            ff = []
            for iii, xxx in zip(ii, xx):
                i0, i1 = self._get_i0_i1(iii)
                ff.append(self.getValueX(xxx, key=key, i0=i0, i1=i1))
            return ff
    
    def getValueX(self, xx: float, key: str, i0: int, i1: int) -> float:

        yy = self.__dict__[key] # the sequence of acquirion data

        X = self.x[i0:i1]
        Y = yy[i0:i1]
        f = interp1d(X, Y, kind='cubic', bounds_error=False, fill_value=(Y[0], Y[-1]))

        return float(f(xx))



class PhysicalSecWall(PhysicalLine):
    
    xf_dict = {
        'mUy': ['min(du/dy)', 0, 0.0],              # position of min(du/dy)
        'S':  ['separation start', [], []],
        'R':  ['reattachment', [], []],
        'T':  ['before separation', [], []]
        # 'P':  ['pleatue', [], []]
    }

    def locate_sep(self, **kwargs):

        _mUy, _sepS, _sepR, _sepT = self._locate_sep(**kwargs)
        for i in range(2):
            self.xf_dict['mUy'][i+1] = _mUy[i]
            try:
                self.xf_dict['S'][i+1] = _sepS[i]
                self.xf_dict['R'][i+1] = _sepR[i]
                self.xf_dict['T'][i+1] = _sepT[i]
            except IndexError:
                pass


    def _locate_sep(self, st: int, ed: int, use_tail: bool = False, x_range: Tuple[float, float] = None):
        
        #* S => separation start position
        #* R => reattachment position
        #* mUy => position of min(du/dy)

        sep_flag = False
        is_sep_flag = False     # whether the separation has happen 
        min_Uy = 1e6

        xx = self.x
        pp = self.Cp
        tau_x = self.dudy

        _mUy   = [0, 0.0]
        _sep_S = [[], []]
        _sep_T = [[], []]
        _sep_R = [[], []]

        for ii in range(st, ed - 1):

            if x_range is not None and (xx[ii] < x_range[0] or xx[ii] > x_range[1]):
                continue
            
            #* find the minimal point of dUdy => mUy
            if tau_x[ii]<min_Uy and tau_x[ii-1]>=tau_x[ii] and tau_x[ii+1]>=tau_x[ii]:
                min_Uy = tau_x[ii]
                _mUy[0] = ii
                _mUy[1] = xx[ii]

            if tau_x[ii]>=0.0 and tau_x[ii+1]<0.0:

                _sep_S[0].append(ii)
                _sep_S[1].append((0.0-tau_x[ii])*(xx[ii+1]-xx[ii])/(tau_x[ii+1]-tau_x[ii])+xx[ii])
                sep_flag = True
                is_sep_flag = True

                # find the pressure before separation
                jj = ii + 1
                while pp[jj] > pp[jj-1]:
                    jj -= 1
                _sep_T[0].append(jj)
                _sep_T[1].append(xx[jj])

            if tau_x[ii]<=0.0 and tau_x[ii+1]>0.0:
                _sep_R[0].append(ii)
                _sep_R[1].append((0.0-tau_x[ii])*(xx[ii+1]-xx[ii])/(tau_x[ii+1]-tau_x[ii])+xx[ii])
                sep_flag = False


        if use_tail and sep_flag:
            _sep_R[0].append(self.n_point-1)
            _sep_R[1].append(xx[-1])

        if use_tail and (not is_sep_flag):
            _sep_S[0].append(self.n_point-1)
            _sep_R[0].append(self.n_point-1)
            _sep_S[1].append(xx[-1])
            _sep_R[1].append(xx[-1])

        return _mUy, _sep_S, _sep_R, _sep_T


class PhysicalSec(PhysicalSecWall):
    '''
    Extracting flow features of a section (features on/near the wall)

    Remark (Yang)
    ---
    static method are used to process field data, while class objuct only accept
    data distribution on the surface
    '''
    _i  = 0     # index of the mesh point
    _X  = 0.0   # location of the feature location
    _value = 0.0

    #* Dictionary of flow features (identify the index and location)
    xf_dict = {
        'Cu':  ['upper crest', _i, _X],             # crest point on upper surface
        'Cl':  ['lower crest', _i, _X],             # crest point on lower surface
        'tu':  ['upper highest', _i, _X],           # highest point on upper surface
        'tl':  ['lower highest', _i, _X],           # lowest point on lower surface
        'tm':  ['max thickness', _i, _X],           # maximum thickness position

        'L': ['upper LE', _i, _X],                  # suction peak near leading edge on upper surface
        'T': ['upper TE', _i, _X],                  # trailing edge upper surface (98% chord length)
        'H': ['upper surface max Ma', _i, _X],      # position of lower upper maximum Mach number
        'S': ['separation start', _i, _X],          # separation start position
        'R': ['reattachment', _i, _X],              # reattachment position
        'Q': ['lower LE', _i, _X],                  # suction peak near leading edge on lower surface
        'M': ['lower surface max Ma', _i, _X],      # position of lower surface maximum Mach number
        'MT': ['lower surface max Ma behind Q', _i, _X],      
        'LD': ['lower surface dent behind Q', _i, _X],
        'mUy': ['min(du/dy)', _i, _X],              # position of min(du/dy)

        'F': ['shock foot', _i, _X],                # shock foot position
        '1': ['shock front', _i, _X],               # shock wave front position
        '3': ['shock hind', _i, _X],                # position of just downstream the shock
        'D': ['dent on plateau', _i, _X],           # largest dent on the suction plateau
        'U': ['local sonic', _i, _X],               # local sonic position
        'B': ['1st dent after L', _i, _X],          # first dent after suction peak [X_L, X_L+0.1]
        #                                           # Note: for weak shock waves, may not reach Mw=1
        #                                           #       define position of U as Mw minimal extreme point after shock foot
        'A': ['maximum Mw after shock', _i, _X],    # maximum wall Mach number after shock wave (or equal to '3')
        'AD': ['minimum Mw between 3 & A', _i, _X],    # (or equal to '3')
        'N': ['new flat boundary', _i, _X],         # starting position of new flat boundary
        #                                           # most of the time, A == N
        'Hi':  ['maximum Hi', _i, _X],              # position of maximum Hi
        'Hc':  ['maximum Hc', _i, _X],              # position of maximum Hc
        '95':  ['95% chord', _i, _X],

        'L1U': ['length 1~U', _value],              # XU-X1
        'L13': ['length 1~3', _value],              # X3-X1
        'LSR': ['length S~R', _value],              # XR-XS
        'lSW': ['single shock', _value],            # single shock wave flag
        'DCp': ['shock strength', _value],          # Cp change through shock wave
        'Err': ['suc Cp area', _value],             # Cp integral of suction plateau fluctuation
        'FSp': ['fluctuation suc-plat', _value],    # Mw fluctuation of suction plateau
        'DMp': ['Mw dent on plateau', _value],      # dMw of Mw dent on suction plateau
        'CLU': ['upper CL', _value],                # CL of upper surface
        'CLL': ['lower CL', _value],                # CL of lower surface
        'CdU': ['upper Cd', _value],                # Cdp of upper surface
        'CdL': ['lower Cd', _value],                # Cdp of lower surface
        'CLw': ['windward CL', _value],             # CL of windward surfaces (before crest point)
        'Cdw': ['windward Cdp', _value],            # Cdp of windward surfaces (before crest point)
        'CLl': ['leeward CL', _value],              # CL of leeward surfaces (behind crest point)
        'Cdl': ['leeward Cdp', _value],             # Cdp of leeward surfaces (behind crest point)
        'kaf': ['slope aft', _value],               # average Mw slope of the aft upper surface (3/N~T)
        'LDdent': ['dent depth between Q and MT', _value]
    }

    def __init__(self, Minf, AoA, Re, Tinf = 460, gamma = 1.40):
        '''
        ### Inputs:
        ```text
        Minf:       Free stream Mach number
        AoA:        Angle of attack (deg)
        Re:         Reynolds number per meter
        ```
        '''
        
        super().__init__()
        self.paras = {'Qratio': 0.2,
                      'Minf':   Minf,
                      'AoA':    AoA,
                      'Re':     Re,
                      'Tinf':   Tinf,
                      'gamma':  gamma}    # the ratio of chord to find suction peak near leading edge on lower surface
        
        self.Tw = None
    
    def setlimdata(self, x, y, Cp, dudy = None):
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.Cp = copy.deepcopy(Cp)
        self.Mw = self.Cp2Mw()
        
        if dudy is not None:
            self.dudy = copy.deepcopy(dudy)
        else:
            self.dudy = None

        iLE = np.argmin(self.x)
        self.iLE = iLE
        self.x -= self.x[iLE]
        self.y -= self.y[iLE]
        self.x[0] = 1.0
        self.x[-1] = 1.0

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')

        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gu  = interp1d(self.x[iLE:],  self.y[iLE:], kind='cubic')
        gl  = interp1d(x_, y_, kind='cubic')
        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)

    def setdata(self, x, y, Cp, Tw, Hi, Hc, dudy):
        '''
        Set the data of this foil or section.

        Data:   ndarray, start from lower surface trailing edge
        '''
        self.setlimdata(x, y, Cp, dudy)
        self.Tw = copy.deepcopy(Tw)
        self.Hi = copy.deepcopy(Hi)
        self.Hc = copy.deepcopy(Hc)

        iLE = np.argmin(self.x)

        fhu = interp1d(self.x[iLE:], self.Hc[iLE:], kind='cubic')
        self.hu = fhu(self.xx)
        
    def set_Mw(self, x, Mw):
        '''
        Set the Mw distribution of this foil or section.

        Data:   ndarray, start from lower surface trailing edge
        '''
        self.x  = copy.deepcopy(x)
        self.Mw = copy.deepcopy(Mw)

        iLE = np.argmin(self.x)
        self.iLE = iLE

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        self.xx = np.arange(0.0, 1.0, 0.001)
        self.mu = fmw(self.xx)

    @staticmethod
    def IsentropicCp(Ma, Minf: float, g=1.4):
        ''' 
        Isentropic flow: Calculate Cp by Mach

        ### Inputs:
        ```text
        Ma:     float, or ndarray
        Minf:   free stream Mach number
        g:      γ=1.4, ratio of the specific heats
        ```
        '''
        X = (2.0+(g-1.0)*Minf**2)/(2.0+(g-1.0)*Ma**2)
        X = X**(g/(g-1.0))
        Cp = 2.0/g/Minf**2*(X-1.0)

        return Cp

    @staticmethod
    def toMw(Cp: np.array, Minf: float, n_ref=100, M_max=2.0):
        '''
        Converting Cp to wall Mach number
        '''
        Ma_ref = np.linspace(0.0, M_max, n_ref)
        Cp_ref = PhysicalSec.IsentropicCp(Ma_ref, Minf)
        f   = interp1d(Cp_ref, Ma_ref, kind='cubic')
        # Cp_ = Cp.copy()
        Cp_ = copy.deepcopy(Cp)
        Cp_ = np.clip(Cp_, Cp_ref[-1], Cp_ref[0])
        return f(Cp_)

    def Cp2Mw(self, n_ref=100, M_max=2.0):
        '''
        Converting Cp to wall Mach number
        '''
        Mw = PhysicalSec.toMw(self.Cp, self.paras['Minf'], n_ref=n_ref, M_max=M_max)

        return Mw

    @staticmethod
    def ShapeFactor(sS, VtS, Tw: float, iUe: int, neglect_error=False):
        '''
        Calculate shape factor Hi & Hc by mesh points on a line pertenticular to the wall.

        ### Inputs:
        ```text
        sS:     ndarray [nMax], distance of mesh points to wall
        VtS:    ndarray [nMax], velocity component of mesh points (parallel to the wall)
        Tw:     wall temperature (K)
        iUe:    index of mesh point locating the outer velocity Ue
        neglect_error:  if True, set shape factor to 0 when error occurs
        ```

        ### Return:
        ```text
        Hi:     incompressible shape factor
        Hc:     compressible shape factor
        ```

        ### Note:
        ```text
        XR  => 物面参考点，考察以 XR 为起点，物面法向 nR 方向上的数据点，共 nMax 个数据点
        sS  => 数据点到物面距离
        VtS => 数据点速度在物面方向的分量

        se:     distance of boundary layer outer boundary to wall
        ds:     𝛿*, displacement thickness
        tt:     θ, momentum loss thickness
        Ue:     outer layer velocity component (parallel to the wall)
        Ue      测试结果显示，直接取最大Ue较为合理，取一定范围内平均，或取固定网格的值，效果不好
        ```
        '''
        nMax= sS.shape[0]
        Ue = VtS[iUe]
        se = sS[iUe]
        ds = 0.0
        tt = 0.0

        if iUe>=nMax or iUe<=int(0.2*nMax):
            if neglect_error:
                return 0.0, 0.0
            else:
                print()
                print('Vts: velocity component of mesh points')
                print(VtS)
                print()
                raise Exception('Error [ShapeFactor]: iUe %d not reasonable (nMax=%d)'%(iUe, nMax))

        for i in range(iUe-1):
            a1 = Ue-VtS[i]
            a2 = Ue-VtS[i+1]
            ds += 0.5*(a1+a2)*(sS[i+1]-sS[i])

        for i in range(iUe-1):
            a1 = VtS[i  ]*(Ue-VtS[i  ])
            a2 = VtS[i+1]*(Ue-VtS[i+1])
            tt += 0.5*(a1+a2)*(sS[i+1]-sS[i])

        Hi = ds/tt*Ue
        Hc = Tw*Hi+Tw-1

        return Hi, Hc

    @staticmethod
    def getHi(X, Y, U, V, T, j0: int, j1: int, nHi: int, neglect_error=False):
        '''
        Calculate shape factor Hi & Hc from field data

        ### Inputs:
        ```text
        Field data: ndarray (nj,nk), X, Y, U, V, T
        j0:     j index of the lower surface TE
        j1:     j index of the upper surface TE
        nHi:    maximum number of mesh points in k direction for boundary layer
        neglect_error:  if True, set shape factor to 0 when error occurs
        ```

        ### Return:
        ```text
        Hi, Hc: ndarray (j1-j0)
        info:   tuple of ndarray (Tw, dudy)
        ```

        ### Note:
        ```text
        Tw:     wall temperature
        dudy:   du/dy
        iUe:    index of mesh point locating the outer velocity Ue
        XR:     reference position on the wall
        ```

        ### Filed data (j,k) index
        ```text
        j: 1  - nj  from far field of lower surface TE to far field of upper surface TE
        j: j0 - j1  from lower surface TE to upper surface TE
        k: 1  - nk  from surface to far field (assuming pertenticular to the wall)
        ```
        '''

        iLE = int(0.5*(j0+j1))
        nj = X.shape[0]
        nk = X.shape[1]
        nn = j1-j0

        Hi = np.zeros(nn)
        Hc = np.zeros(nn)
        Tw = np.zeros(nn)
        dudy = np.zeros(nn)

        #* Locate boundary layer edge index iUe & calculate du/dy
        sS  = np.zeros([nn,nHi])
        VtS = np.zeros([nn,nHi])
        iUe = np.zeros(nn, dtype=int)

        for j in range(nn):
            jj = j0+j
            XR = np.array([X[jj,0], Y[jj,0]])
            tR = np.array([X[jj+1,0]-X[jj-1,0], Y[jj+1,0]-Y[jj-1,0]])
            tR = tR/np.linalg.norm(tR)
            if tR[0]<0.0:
                tR = -tR

            for i in range(nHi-1):
                XS = np.array([X[jj,i+1], Y[jj,i+1]])
                VS = np.array([U[jj,i+1], V[jj,i+1]])

                sS [j,i+1] = np.linalg.norm(XR-XS)
                VtS[j,i+1] = np.dot(tR,VS)

            iUe[j]  = np.argmax(np.abs(VtS[j,:]))
            dudy[j] = VtS[j,1]/sS[j,1]
            Tw[j]   = T[jj,0]
        
        #* Smooth iUe at shock wave foot
        nspan = 4
        for j in range(nn-2*nspan):
            jj = j+nspan
            r1 = 0.5*(iUe[jj-nspan]+iUe[jj+nspan])
            r2 = abs(iUe[jj+nspan]-iUe[jj-nspan])
            r3 = abs(iUe[jj]-iUe[jj-nspan]) + abs(iUe[jj]-iUe[jj+nspan])
            if r3>r2:
                iUe[jj] = int(r1)

        #* Calculate Hi & Hc
        for j in range(nn):
            Hi[j], Hc[j] = PhysicalSec.ShapeFactor(sS[j,:], VtS[j,:], 
                            Tw[j], iUe[j], neglect_error=neglect_error)

        #* Limit leading edge Hi
        r1 = 1.0
        r2 = 1.0
        r3 = 1.0
        r4 = 1.0
        for j in range(nn):
            jj = j0+j
            if (X[jj,0]-0.05)*(X[jj+1,0]-0.05)<=0.0 and jj<iLE:
                r1 = Hi[j]
                r3 = Hc[j]
            if (X[jj,0]-0.05)*(X[jj+1,0]-0.05)<=0.0 and jj>=iLE:
                r2 = Hi[j]
                r4 = Hc[j]

        for j in range(nn):
            jj = j0+j
            if X[jj,0]<0.05 and jj<iLE:
                Hi[j] = r1
                Hc[j] = r3
            if X[jj,0]<0.05 and jj>=iLE:
                Hi[j] = r2
                Hc[j] = r4

        return Hi, Hc, (Tw, dudy)

    
    def get_force_moment(self, X, Y, U, V, T, P, j0: int, j1: int, x_mc = 0.25, y_mc = 0.0):
        _paras = copy.deepcopy(self.paras)
        _paras['x_mc'] = x_mc
        _paras['y_mc'] = y_mc
        return PhysicalSec.get_force(X, Y, U, V, T, P, j0, j1, _paras)

    @staticmethod
    def get_force(X, Y, U, V, T, P, j0: int, j1: int, paras, ptype='Cp'):
        '''
        Calculate cl and cd from field data

        ### Inputs:
        ```text
        Field data: X, Y, U, V, T, P
            - in ndarray (nj,nk) type
            - data should be at nodes, rather than at cell center (cfl3d -> .prt are nodes value)
        j0:     j index of the lower surface TE node
        j1:     j index of the upper surface TE node
        paras:  'gamma'    : self.gamma,
                'Minf'     : self.Minf,
                'Re'       : self.Re,
                'Tinf'     : self.Tinf,
                'AoA'      : self.AoA
        ```

        ### Return:
        ```text
        cx, cy: force coefficient of x,y dir
        cl, cd: lift coef. and drag coef.
        ```

        ### Note:

        ### Filed data (j,k) index
        ```text
        j: 1  - nj  from far field of lower surface TE to far field of upper surface TE
        j: j0 - j1  from lower surface TE to upper surface TE
        k: 1  - nk  from surface to far field (assuming pertenticular to the wall)
        '''

        cx = 0.0
        cy = 0.0
        cmz = 0.0
        # print(self.Minf, self.Re, self.Tinf)

        for j in range(j0, j1-1):
            
            point1 = np.array([X[j, 0], Y[j, 0]])        # coordinate of surface point j
            point2 = np.array([X[j, 1], Y[j, 1]]) 
            point3 = np.array([X[j + 1, 0], Y[j + 1, 0]])
            point4 = np.array([X[j + 1, 1], Y[j + 1, 1]])
            
            p_cen = 0.25 * (P[j, 0] + P[j, 1] + P[j+1, 0] + P[j+1, 1])
            t_cen = 0.25 * (T[j, 0] + T[j, 1] + T[j+1, 0] + T[j+1, 1])
            ### u,v on wall kepp origin
            # u_cen = 0.25 * (U[j, 0] + U[j, 1] + U[j+1, 0] + U[j+1, 1])
            # v_cen = 0.25 * (V[j, 0] + V[j, 1] + V[j+1, 0] + V[j+1, 1])
            ### u,v on wall set to 0
            u_cen = 0.5 * (U[j, 1] + U[j+1, 1])
            v_cen = 0.5 * (V[j, 1] + V[j+1, 1])
            
            vec_sl = point3 - point1                    # surface vector sl
            veclen = np.linalg.norm(vec_sl)   # length of surface vector sl
            vec_sl = vec_sl / veclen
            area = 0.5 * np.linalg.norm(np.cross(point4 - point1, point3 - point2))
            
            metrix_nt2xy = np.array([[vec_sl[0], vec_sl[1]],[-vec_sl[1], vec_sl[0]]])

            # pressure part, normal to wall(sl)
            ### P
            if ptype == 'P':
                dfp_n = 1.43 / (paras['gamma'] * paras['Minf']**2) * (paras['gamma'] * p_cen - 1) * veclen
            else:
                dfp_n = p_cen * veclen

            # viscous part, tang to wall(sl)
            mu = t_cen**1.5 * (1 + 198.6 / paras['Tinf']) / (t_cen + 198.6 / paras['Tinf'])
            dfv_t = 0.063 / (paras['Minf'] * paras['Re']) * mu * np.dot(np.array([u_cen, v_cen]), vec_sl) * veclen**2 / area

            # print(mu, t_cen, p_cen, np.array([u_cen,v_cen]))
            dfp = np.dot(np.array([dfv_t, -dfp_n]), metrix_nt2xy)

            cx += dfp[0]
            cy += dfp[1]

            sl_cen = 0.5 * (point1 + point3)
            cmz += dfp[1] * (sl_cen[0] - paras['x_mc']) - dfp[0] * (sl_cen[1] - paras['y_mc'])


        metrix_xy2ab = np.array([[ut.cos(paras['AoA']), -ut.sin(paras['AoA'])],[ut.sin(paras['AoA']), ut.cos(paras['AoA'])]])

        fld = np.dot(np.array([cx, cy]), metrix_xy2ab)

        return cx, cy, fld[1], fld[0], cmz

    def _get_i0_i1(self, ii):
        if ii >= self.iLE:
            i0 = max(self.iLE, ii-4)
            i1 = min(i0 + 7, len(self.x))
        else:
            i1 = min(self.iLE, ii+4)
            i0 = max(i1 - 7, 0)

        return i0, i1 

    #!: locate the position of flow features
    def locate_basic(self, dMwcri_L=1.0):
        '''
        Locate the index and position of basic flow features.

        ### Get value of: L, T, Q, M
        '''
        X = self.x
        M = self.Mw

        nn  = X.shape[0]
        iLE = self.iLE

        #!: Basic features
        #* L => suction peak near leading edge on upper surface
        # 1: maximum extreme point
        # 2: dMw/dx = 1
        i_L = 0
        for i in range(int(0.25*nn)):
            ii = i + iLE
            if X[ii] > 0.2:
                break
            if M[ii-1]<=M[ii] and M[ii]>=M[ii+1]:
                i_L = ii
                break
    
        if i_L == 0:
            dMw2 = 0.0
            for i in range(int(0.25*nn)):
                ii = i + iLE+1
                dMw1 = dMw2
                dMw2 = (M[ii+1]-M[ii])/(X[ii+1]-X[ii])
                if dMw1>=dMwcri_L and dMw2<dMwcri_L:
                    i_L = ii
                    break

        self.xf_dict['L'][1] = i_L
        self.xf_dict['L'][2] = X[i_L]

        #* T => trailing edge upper surface (98% chord length)
        for i in range(int(0.2*nn)):
            ii = nn-i-1
            if X[ii]<=0.95 and X[ii+1]>0.95:
                self.xf_dict['95'][1] = ii
                self.xf_dict['95'][2] = 0.95
                break

            if X[ii]<=0.98 and X[ii+1]>0.98:
                self.xf_dict['T'][1] = ii
                self.xf_dict['T'][2] = 0.98
        
        #* H => position of upper surface maximum Mach number
        i_H = 0
        max1 = -1.0
        for i in np.arange(iLE, nn-2, 1):
            if M[i-1]<=M[i] and M[i+1]<=M[i] and M[i]>max1:
                max1 = M[i]
                i_H = i

        self.xf_dict['H'][1] = i_H
        self.xf_dict['H'][2] = X[i_H]

        #* Q => suction peak near leading edge on lower surface
        for ii in range(iLE - 1, 1, -1):
            if X[ii] > self.paras['Qratio']:
                continue
            if M[ii-1]<=M[ii] and M[ii]>=M[ii+1]:
                self.xf_dict['Q'][1] = ii
                self.xf_dict['Q'][2] = X[ii]
                break
        else:
            self.xf_dict['Q'][1] = iLE
            self.xf_dict['Q'][2] = 0.0

        i_Q = self.xf_dict['Q'][1]

        #* MT => position of lower surface maximum Mach number behund suction peak
        # if not exist, set i_MT to i_Q
        i_MT = i_Q
        max1 = -1.0
        for i in range(1, i_Q - 1):
            if M[i-1]<=M[i] and M[i+1]<=M[i] and M[i]>max1:
                max1 = M[i]
                i_MT = i

        self.xf_dict['MT'][1] = i_MT
        self.xf_dict['MT'][2] = X[i_MT]
                        
        #* M => position of lower surface maximum Mach number
        i_M = i_MT if M[i_MT] > M[i_Q] else i_Q
        self.xf_dict['M'][1] = i_M
        self.xf_dict['M'][2] = X[i_M]
                
        #* LD => position of lowest Ma between lower surface suction peak and max Ma at lower surface
        if i_Q == iLE or i_Q == i_MT:
            # no lower surface suction peak
            self.xf_dict['LD'][1] = iLE
            self.xf_dict['LD'][2] = 0.
            self.xf_dict['LDdent'][1] = 0.
        else:
            i_LD = self.xf_dict['MT'][1]
            min1 = 2.0
            for _i in range(i_MT + 1, i_Q):
                if M[_i-1]>=M[_i] and M[_i+1]>=M[_i] and M[_i]<min1:
                    min1 = M[_i]
                    i_LD = _i
            
            self.xf_dict['LD'][1] = i_LD
            self.xf_dict['LD'][2] = X[i_LD]
            self.xf_dict['LDdent'][1] = M[i_MT] + (X[i_LD] - X[i_MT]) / (X[i_Q] - X[i_MT]) * (M[i_Q] - M[i_MT]) - M[i_LD]
            
    def locate_sep(self):
        '''
        Locate the index and position of flow features about du/dy.

        ### Get value of: S, R, mUy
        '''
        # super().locate_sep(self.iLE, self.x.shape[0], x_range=(0.0, 1.0))
        _mUy, _sepS, _sepR, _sepT = self._locate_sep(self.iLE, self.x.shape[0], x_range=(0.1, 0.9))
        # print(_mUy, _sepS, _sepR, _sepT)
        for i in range(2):
            self.xf_dict['mUy'][i+1] = _mUy[i]
            try:
                self.xf_dict['S'][i+1] = _sepS[i][0]
                self.xf_dict['R'][i+1] = _sepR[i][0]
                self.xf_dict['T'][i+1] = _sepT[i][0]
            except IndexError:
                pass

    def locate_geo(self):
        '''
        Locate the index and position of geometry related flow features.\n

        ### Get value of: Cu, Cl, tu, tl, tm
        '''
        X  = self.x
        xx = self.xx
        yu = self.yu
        yl = self.yl
        iLE = self.iLE
        n0 = xx.shape[0]

        #* tm => maximum thickness
        #* tu => highest point on upper surface
        #* tl => lowest point on lower surface
        x_max = xx[np.argmax(yu-yl)]
        x_mu  = xx[np.argmax(yu)]
        x_ml  = xx[np.argmin(yl)]

        self.xf_dict['tm'][1] = np.argmin(np.abs(X[iLE:]-x_max)) + iLE
        self.xf_dict['tm'][2] = x_max
        self.xf_dict['tu'][1] = np.argmin(np.abs(X[iLE:]-x_mu )) + iLE
        self.xf_dict['tu'][2] = x_mu
        self.xf_dict['tl'][1] = np.argmin(np.abs(X[:iLE]-x_ml ))
        self.xf_dict['tl'][2] = x_ml

        #* Cu => crest point on upper surface
        aa = self.paras['AoA']/180.0*np.pi
        x0 = np.array([0.0, 0.0])
        x1 = np.array([np.cos(aa), np.sin(aa)])

        ds = np.zeros(n0)
        for i in range(n0):
            xt = np.array([xx[i], yu[i]])
            if xx[i] > 0.9:
                continue
            ds[i], _ = ratio_vec(x0, x1, xt)
        ii = np.argmax(ds)

        self.xf_dict['Cu'][1] = np.argmin(np.abs(X[iLE:]-xx[ii])) + iLE
        self.xf_dict['Cu'][2] = xx[ii]

        #* Cl => crest point on lower surface
        ds = np.zeros(n0)
        for i in range(n0):
            if xx[i] > 0.9:
                continue
            xt = np.array([xx[i], yl[i]])
            ds[i], _ = ratio_vec(x0, x1, xt)
        ii = np.argmax(ds)

        self.xf_dict['Cl'][1] = np.argmin(np.abs(X[:iLE]-xx[ii]))
        self.xf_dict['Cl'][2] = xx[ii]

    def locate_shock(self, dMwcri_1=-1.0, info=False):
        '''
        Locate the index and position of shock wave related flow features.

        ### Get value of: 1, 3, F, U, D, A, B
        
        ### Inputs:
        ```text
        dMwcri_1: critical value locating shock wave front
        ```
        '''
        X   = self.x        # [n]
        xx  = self.xx       # [1000]
        mu  = self.mu       # [1000]
        nn  = xx.shape[0]
        iLE = self.iLE

        dMw = np.zeros(nn)
        for i in range(nn-1):
            if xx[i]<=0.02:
                continue
            if xx[i]>=0.98:
                continue
            dMw[i] = (mu[i + 1] - mu[i]) / (xx[i + 1] - xx[i])
            dMw[i] = min(dMw[i], 2)

        d2Mw = np.zeros(nn)
        for i in range(nn-1):
            if xx[i]<0.02 or xx[i]>0.95:
                continue
            
            #d2Mw[i] = (dMw[i+2]+dMw[i+1]-dMw[i]-dMw[i-1])/2/(xx[i+1]-xx[i-1])
            #d2Mw[i] = (dMw[i+1]-dMw[i-1])/(xx[i+1]-xx[i-1])
            d2Mw[i] = (0.5*dMw[i+7]+0.5*dMw[i+4]+2*dMw[i+1]-
                        2*dMw[i]-0.5*dMw[i-3]-0.5*dMw[i-6])/4.5/(xx[i+1]-xx[i-1])

        #* Check shock and shock properties
        flag, i_F, i_1, i_U, i_3 = PhysicalSec.check_singleshock(xx, mu, dMw, d2Mw, dMwcri_1, info=info)

        self.xf_dict['lSW'][1] = flag

        #* F => shock foot position
        self.xf_dict['F'][1] = np.argmin(np.abs(X[iLE:]-xx[i_F])) + iLE
        self.xf_dict['F'][2] = xx[i_F]

        #* 1 => shock wave front position
        self.xf_dict['1'][1] = np.argmin(np.abs(X[iLE:]-xx[i_1])) + iLE
        self.xf_dict['1'][2] = xx[i_1]

        #* 3 => position of just downstream the shock
        self.xf_dict['3'][1] = np.argmin(np.abs(X[iLE:]-xx[i_3])) + iLE
        self.xf_dict['3'][2] = xx[i_3]
        #* U => local sonic position
        self.xf_dict['U'][1] = np.argmin(np.abs(X[iLE:]-xx[i_U])) + iLE
        self.xf_dict['U'][2] = xx[i_U]

        #* D => dent on the suction plateau
        # maximum (linear Mw - actual Mw) between L and 1
        x_1 = self.xf_dict['1'][2]
        x_L = max(self.xf_dict['L'][2], 0.05)
        m_1 = self.getValue('1','Mw')
        m_L = self.getValue('L','Mw')
        lL1 = x_1-x_L
        i_D = 0
        min_D = 0.0
        for i in np.arange(2, i_1-1, 1):

            if xx[i]<x_L:
                continue

            tt = (xx[i]-x_L)/lL1
            ss = (1-tt)*m_L + tt*m_1
            dM = ss - mu[i]

            if dM > min_D:
                i_D = i
                min_D = dM

        if i_D==0:
            self.xf_dict['D'][1] = self.xf_dict['L'][1]
            self.xf_dict['D'][2] = self.xf_dict['L'][2]
        else:
            self.xf_dict['D'][1] = np.argmin(np.abs(X[iLE:]-xx[i_D])) + iLE
            self.xf_dict['D'][2] = xx[i_D]

        #* B => first dent after suction peak [X_L, X_L+0.1]
        # minimum Mw between L and L+0.1
        x_L = self.xf_dict['L'][2]
        i_B = 0
        for i in np.arange(2, i_1-1, 1):

            if xx[i]<x_L or xx[i]>x_L+0.1:
                continue

            if mu[i-1]>=mu[i] and mu[i]<=mu[i+1] and i_B==0:
                i_B = i

        if i_B == 0:
            self.xf_dict['B'][1] = self.xf_dict['L'][1]
            self.xf_dict['B'][2] = self.xf_dict['L'][2]
        else:
            self.xf_dict['B'][1] = np.argmin(np.abs(X[iLE:]-xx[i_B])) + iLE
            self.xf_dict['B'][2] = xx[i_B]

        #* A => maximum Mw after shock
        # Find the maximum position of Mw in range [x_3, 0.9]
        i_A = 0
        max_A = 0.0
        for i in range(i_3, nn-1):
            if xx[i]>0.9:
                break
            if mu[i]>max_A:
                i_A = i
                max_A = mu[i]
            elif mu[i]>=mu[i_3]*0.8 and mu[i]>mu[i-1] and mu[i]>mu[i+1]:
                i_A = i

        x_A = xx[i_A]
        self.xf_dict['A'][1] = np.argmin(np.abs(X[iLE:]-x_A)) + iLE
        self.xf_dict['A'][2] = x_A
        
        #* AD => local minium between 3 and A
        i_AD = i_3
        min_AD = 10.0
        for _i in range(i_3, i_A):
            if mu[_i] < mu[_i-1] and mu[_i] > mu[_i+1] and mu[_i] < min_AD:
                i_AD = _i
                
        x_AD = xx[i_AD]
        self.xf_dict['AD'][1] = np.argmin(np.abs(X[iLE:]-x_AD)) + iLE
        self.xf_dict['AD'][2] = x_AD
        
        return i_1

    def locate_BL(self, i_1):
        '''
        Locate the index and position of boundary layer related flow features. \n
        
        i-1: index of shock wave front position in self.xx

        ### Get value of: N, Hi, Hc
        '''
        X   = self.x
        xx  = self.xx
        hu  = self.hu
        nn  = xx.shape[0]
        iLE = self.iLE

        #* Hi, Hc => position of maximum Hi, Hc after shock wave front
        # For cases when shock wave is weak, and Hc just keeps growing, set 0
        i_H = 0
        max1 = 0.0
        for i in np.arange(i_1, nn-2, 1):

            if xx[i] > 0.95:
                break

            if hu[i-1]<=hu[i] and hu[i+1]<=hu[i] and hu[i]>max1:
                max1 = hu[i]
                i_H = i
                
        x_H = xx[i_H]
        self.xf_dict['Hc'][1] = np.argmin(np.abs(X[iLE:]-x_H)) + iLE
        self.xf_dict['Hc'][2] = x_H
        self.xf_dict['Hi'][1] = self.xf_dict['Hc'][1]
        self.xf_dict['Hi'][2] = x_H

        #* N => starting position of new flat boundary
        # i.e., position of minimum Hc after shock wave front
        # For cases when shock wave is weak, and Hc just keeps growing, set 0
        i_N = 0
        min1 = 1000.0
        for i in np.arange(i_1, nn-1, 1):
            if hu[i-1]>=hu[i] and hu[i+1]<=hu[i] and hu[i]<min1:
                min1 = hu[i]
                i_N = i

        x_N = xx[i_N]
        self.xf_dict['N'][1] = np.argmin(np.abs(X[iLE:]-x_N)) + iLE
        self.xf_dict['N'][2] = x_N

    @staticmethod
    def shock_property(xu, mu, dMw, d2Mw, dMwcri_1):
        '''
        >>> i_F, i_1, i_U, i_3 = shock_property(xu, mu, dMw, d2Mw, dMwcri_1)

        ### Return:
        ```text
        Index of xu for: F, 1, U, 3
        ```
        '''
        nn  = xu.shape[0]

        #* F => shock foot position (largest dMw/dx -> largest decsending position of Mw)
        i_F = np.argmin(dMw)
        x_F = xu[i_F]

        #* 1 => shock wave front position
        # Find the kink position of dMw in range [x_F-0.2, x_F], defined as dMw = -1
        i_1 = 0
        i_cri = 0
        i_md2 = 0
        for i in np.arange(i_F, 1, -1):

            # 1. Within the range of [x_F-0.2, x_F]
            if xu[i]<x_F-0.2:
                break

            # 2. Locate dMw = dMwcri_1 (tend to go too much upstream)
            if dMw[i]>=dMwcri_1 and dMw[i+1]<dMwcri_1 and i_cri==0:
                i_cri = i

            # 3. Locate min d2Mw/dx2 (tend to go too much downstream)
            if d2Mw[i]<=d2Mw[i-1] and d2Mw[i]>d2Mw[i+1] and i_md2==0:
                i_md2 = i
        
        if i_md2-i_cri > 2*(i_F-i_md2):
            i_1 = i_md2
        elif 2*(i_md2-i_cri) < i_F-i_md2:
            i_1 = i_cri
        else:
            i_1 = int(0.5*(i_cri+i_md2))

        '''
        print(i_cri, i_md2, i_F, xu[i_cri], xu[i_md2], dMw[i_md2], dMw[i_F])

        import matplotlib.pyplot as plt
        plt.plot(xu, mu, 'b')
        plt.plot(xu, d2Mw/1000, 'r')
        plt.plot([xu[i_cri], xu[i_md2]], [mu[i_cri], mu[i_md2]], 'bo')
        plt.plot([xu[i_1]], [mu[i_1]], 'ro')
        plt.show()
        '''

        #* 3 => position of just downstream the shock
        # Find the first flat position of Mw in range [x_F, x_F+0.2], defined as dMw = 0 or -1
        i_3 = 0
        i_cri = 0
        i_md2 = 0
        i_flat = 0
        for i in np.arange(i_F, nn-1, 1):

            # 1. Within the range of [x_F, x_F+0.2]
            if xu[i]>x_F+0.2:
                break

            # 2. Locate dMw = dMwcri_1 (tend to go too much downstream)
            if dMw[i]<=dMwcri_1 and dMw[i+1]>dMwcri_1 and i_cri==0:
                i_cri = i

            # 3. Locate min d2Mw/dx2 (tend to go too much upstream)
            if d2Mw[i]<=d2Mw[i-1] and d2Mw[i]>d2Mw[i+1] and i_md2==0:
                i_md2 = i

            # 4. Locate the first flat position of Mw
            if dMw[i]<=0.0 and dMw[i+1]>0.0:
                i_flat = i

        if i_flat!=0 and i_flat-i_F < 2*(i_cri-i_F):
            i_3 = i_flat
        elif i_cri-i_md2 > 2*(i_md2-i_F):
            i_3 = i_md2
        elif 2*(i_cri-i_md2) < i_md2-i_F:
            i_3 = i_cri
        else:
            i_3 = int(0.5*(i_cri+i_md2))

        '''
        print('F     %3d  %.2f'%(i_F,   xu[i_F]))
        print('d2Mw  %3d  %.2f'%(i_md2, xu[i_md2]))
        print('cri   %3d  %.2f'%(i_cri, xu[i_cri]))
        print('dMw=0 %3d  %.2f'%(i_flat,xu[i_flat]))
        print('3     %3d  %.2f'%(i_3,   xu[i_3]))
        print()
        '''

        #* U => local sonic position
        i_U = 0
        for i in np.arange(i_1, i_3, 1):
            if mu[i]>=1.0 and mu[i+1]<1.0:
                i_U = i
                break
        
        #* Neglect small Mw bump near leading edge
        if xu[i_1]<0.1 and mu[i_1]<1.10:
            i_1=0; i_U=0; i_3=0

        return i_F, i_1, i_U, i_3

    @staticmethod
    def check_singleshock(xu, mu, dMw, d2Mw, dMwcri_1, info=False):
        '''
        Check whether is single shock wave or not

        >>> flag, i_F, i_1, i_U, i_3 = check_singleshock(xu, mu, dMw, d2Mw, dMwcri_1)

        ### Inputs:
        ```text
        xu:     ndarray, x location
        mu:     ndarray, wall Mach number of upper surface
        dMw:    ndarray, slope of wall Mach number
        dMwcri_1: critical value locating shock wave front
        ```

        ### flag: 
        ```text
         1: single shock wave
         0: shockless
        -1: multiple shock waves 
        ```
        '''
        #* find the ma = 1 after first shock wave
        _flag_first = False
        for _i in range(1, len(mu)-1):
            if _flag_first and mu[_i] >= 1.:
                break
            if mu[_i] > 1. and mu[_i + 1] < 1.:
                _flag_first = True

        #* Get 1st shock
        i_F, i_1, i_U, i_3 = PhysicalSec.shock_property(xu[:_i], mu[:_i], dMw[:_i], d2Mw[:_i], dMwcri_1)
        d_F = dMw[i_F]

        #* Check if shockless
        # Check if Mw jump exists and M1>1.0
        if d_F>dMwcri_1 or mu[i_1]<1.0 or i_1==0:
            if info:
                print('  Shockless:    XF=%.2f MF=%.2f dM/dX=%.2f'%(xu[i_F], mu[i_F], d_F))
            return 0, i_F, 0, 0, 0

        #* Check if 2nd shock wave exists
        # Remove first shock
        dm  = dMw.copy()
        d2m = d2Mw.copy()
        nn  = xu.shape[0]
        for i in np.arange(i_F, nn, 1, dtype=int):
            if dm[i]<=0.0:
                dm[i]=0.0
                d2m[i]=0.0
            else:
                break
        for i in np.arange(i_F, 0, -1, dtype=int):
            if dm[i]<=0.0:
                dm[i]=0.0
                d2m[i]=0.0
            else:
                break
        
        # Locate second shock
        dMwcri_F = max(dMwcri_1, 0.5*d_F)
        _iF, _i1, _iU, _i3 = PhysicalSec.shock_property(xu, mu, dm, d2m, dMwcri_1)
        if dm[_iF]<dMwcri_F and _i1!=0 and _i3!=0:
            # Locate sharp change of Mw

            if mu[_i1]>1.0 and mu[_i3]<1.05:
                # Check supersonic wave front and 'subsonic' wave hind
                if info:
                    print('  Second shock: X1=%.2f M1=%.2f M2=%.2f'%(xu[_i1], mu[_i1], mu[_i3]))
                return -1, i_F, i_1, i_U, i_3

        return 1, i_F, i_1, i_U, i_3

    def aux_features(self):
        '''
        Calculate auxiliary features based on basic, geo, and shock features.

        ### Get value of: Lengths, lSW, DCp, Err, DMp, FSp
        '''
        X  = self.x
        Y  = self.y
        x1 = self.xf_dict['1'][2]
        n0 = len(X)
        
        self.xf_dict['L1U'][1] = self.xf_dict['U'][2] - x1
        self.xf_dict['L13'][1] = self.xf_dict['3'][2] - x1
        self.xf_dict['LSR'][1] = self.xf_dict['R'][2] - self.xf_dict['S'][2]
        self.xf_dict['DCp'][1] = self.getValue('3','Cp') - self.getValue('1','Cp')

        cosA = np.cos(self.paras['AoA']/180.0*np.pi)
        sinA = np.sin(self.paras['AoA']/180.0*np.pi)
        #* Err => Cp integral of suction plateau fluctuation
        #* DMp => Mw dent on suction plateau
        #* FSp => Mw fluctuation of suction plateau
        # If can not find suction peak, err = 0, DMp = 0.0, FSp = 0.0
        Err = 0.0
        DMp = 0.0
        FSp = 0.0
        iL  = self.xf_dict['L'][1]
        if iL!=0:
            i1 = self.xf_dict['1'][1]
            xL  = self.xf_dict['L'][2]

            Cp0 = np.array([xL, self.getValue('L','Cp')])
            Cp1 = np.array([x1, self.getValue('1','Cp')])

            Mw0 = self.getValue('L','Mw')
            Mw1 = self.getValue('1','Mw')
            lL1 = x1-xL
            bump_ = 0.0
            dent_ = 0.0

            for i in np.arange(iL, i1, 1):

                vec = np.array([X[i], self.Cp[i]])
                s, _ = ratio_vec(Cp0, Cp1, vec)
                Err += s*(X[i+1]-X[i])

                tt = (X[i]-xL)/lL1
                ss = (1-tt)*Mw0 + tt*Mw1
                DMp = max(DMp, ss-self.Mw[i])

                local_avg_mw = (self.Mw[i-2]+self.Mw[i]+self.Mw[i+2])/3.0

                if self.Mw[i-4]>=local_avg_mw and local_avg_mw<=self.Mw[i+4] and dent_<=0.0:
                    if bump_>0.0:
                        FSp += bump_ - local_avg_mw
                    dent_ = local_avg_mw
                    bump_ = 0.0
                elif self.Mw[i-4]<=local_avg_mw and local_avg_mw>=self.Mw[i+4] and bump_<=0.0:
                    if dent_>0.0:
                        FSp += local_avg_mw - dent_
                    bump_ = local_avg_mw
                    dent_ = 0.0

        self.xf_dict['Err'][1] = abs(Err)*cosA
        self.xf_dict['DMp'][1] = DMp
        self.xf_dict['FSp'][1] = FSp
    
    def aux_features_BL(self):
        '''
        get secondary features from BL
        
        ### Get value of: kaf, CLU, CLL, CLw, Cdw, CLl, Cdl
        '''
        
        X  = self.x
        Y  = self.y
        n0 = len(X)
        
        cosA = np.cos(self.paras['AoA']/180.0*np.pi)
        sinA = np.sin(self.paras['AoA']/180.0*np.pi)

        #* kaf => average Mw slope of the aft upper surface (3/N~T)
        xN  = self.xf_dict['N'][2]
        mN  = self.getValue('N','Mw')
        xT  = self.xf_dict['T'][2]
        mT  = self.getValue('T','Mw')
        if xN < 0.1:
            xN = self.xf_dict['3'][2]
            mN = self.getValue('3','Mw')

        self.xf_dict['kaf'][1] = (mT-mN)/(xT-xN)

        #* CLU => CL of upper surface
        # wall vector = [dx,dy]
        # outward wall vector = [-dy,dx]
        # outward pressure force vector = Cp*[dy,-dx]
        PFy = 0.0   # y direction pressure force
        PFx = 0.0   # x direction pressure force

        for i in np.arange(self.iLE, n0-1, 1):
            Cp_ = 0.5*(self.Cp[i]+self.Cp[i+1])
            PFx += Cp_*(Y[i+1]-Y[i])
            PFy += Cp_*(X[i]-X[i+1])
        self.xf_dict['CLU'][1] = PFy*cosA - PFx*sinA
        self.xf_dict['CdU'][1] = PFy*sinA + PFx*cosA
        
        PFx = 0.0; PFy = 0.0
        for i in np.arange(0, self.iLE, 1):
            Cp_ = 0.5*(self.Cp[i]+self.Cp[i+1])
            PFx += Cp_*(Y[i+1]-Y[i])
            PFy += Cp_*(X[i]-X[i+1])
        self.xf_dict['CLL'][1] = PFy*cosA - PFx*sinA
        self.xf_dict['CdL'][1] = PFy*sinA + PFx*cosA

        #* Windward and leeward pressure force (CL, Cdp)
        icu = self.xf_dict['Cu'][1]
        icl = self.xf_dict['Cl'][1]

        PFx = 0.0; PFy = 0.0
        for i in np.arange(0, icl, 1):          # Leeward (lower surface)
            Cp_  = 0.5*(self.Cp[i]+self.Cp[i+1])
            PFx += Cp_*(Y[i+1]-Y[i])
            PFy += Cp_*(X[i]-X[i+1])
        for i in np.arange(icu, n0-1, 1):       # Leeward (upper surface)
            Cp_  = 0.5*(self.Cp[i]+self.Cp[i+1])
            PFx += Cp_*(Y[i+1]-Y[i])
            PFy += Cp_*(X[i]-X[i+1])
        self.xf_dict['CLl'][1] = PFy*cosA - PFx*sinA
        self.xf_dict['Cdl'][1] = PFy*sinA + PFx*cosA

        PFx = 0.0; PFy = 0.0
        for i in np.arange(icl, icu, 1):        # Windward
            Cp_ = 0.5*(self.Cp[i]+self.Cp[i+1])
            PFx += Cp_*(Y[i+1]-Y[i])
            PFy += Cp_*(X[i]-X[i+1])
        self.xf_dict['CLw'][1] = PFy*cosA - PFx*sinA
        self.xf_dict['Cdw'][1] = PFy*sinA + PFx*cosA

    def extract_features(self, info=False, is_basic=2):
        '''
        Extract flow features list in the dictionary.
        
        ### para
        - `is_basic`: if is set 0, only basic and shock features are extracted
            - need only geometry and cp
        '''
        # rely only on cp (Mw)
        self.locate_basic()
        self.locate_geo()
        i_1 = self.locate_shock(info=info)
        
        # rely on cf
        if is_basic > 0:
            if self.dudy is not None:
                self.locate_sep()
                self.aux_features()
            else:
                print('WARNING: dudy is not assigned')
        
        if is_basic > 1:
            if i_1 > 0 and self.Tw is not None:
                self.locate_BL(i_1)
                self.aux_features_BL()
            else:
                print('WARNING: Tw is not assigned or not single shock wave')

    #!: output features
    def output_features(self, fname="feature2d.txt", append=True, keys_=None):
        '''
        Output all features to file.

        ### Inputs:
        ```text
        keys:  list of key strings for output. None means default.
        ```

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        value or ('X', 'Cp', 'Mw', 'Tw', 'Hi', 'Hc', 'dudy')
        ```
        '''
        if keys_ is not None:
            keys = copy.deepcopy(keys_)
        else:
            keys = ['X', 'Mw', 'Hc']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in self.xf_dict.keys():
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


class PhysicalXfoil(PhysicalSec):
    '''
    Extract features from Xfoil (low speed) results.
    '''
    def __init__(self, Minf, AoA, Re):
        super().__init__(Minf, AoA, Re)

    def setdata(self, x, y, Cp):
        '''
        Set the data of this foil or section.

        x,y,Cp: list, start from upper surface trailing edge (order from xfoil)
        '''
        x_  = list(reversed(x))
        y_  = list(reversed(y))
        Cp_ = list(reversed(Cp))

        n = int(len(x_)/2)
        x_  = x_[:n] + [0.0] + x_[n:]
        y_  = y_[:n] + [0.5*(y_[n]+y_[n-1])] + y_[n:]
        Cp_ = Cp_[:n] + [0.5*(Cp_[n]+Cp_[n-1])] + Cp_[n:]

        self.x = np.array(x_)
        self.y = np.array(y_)
        self.Cp = np.array(Cp_)
        self.Mw = self.Cp2Mw()

        iLE = np.argmin(self.x)
        
        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        gu  = interp1d(self.x[iLE:], self.y [iLE:], kind='cubic')
        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gl  = interp1d(x_, y_, kind='cubic')

        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)

        self.iLE = iLE

    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_geo()

    def output_features(self, fname="feature-xfoil.txt", append=True):
        '''
        Output all features to file.

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        'X', 'Mw', 'Cp'
        ```
        '''
        keys = ['X','Mw','Cp']
        features = ['L', 'T', 'Q', 'M', 'Cu', 'Cl', 'tu', 'tl', 'tm']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in features:
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


class PhysicalTSFoil(PhysicalSec):
    '''
    Extract features from pyTSFoil (transonic speed) results.
    '''
    def __init__(self, Minf, AoA, Re):
        super().__init__(Minf, AoA, Re)

    def setdata(self, xu, yu, xl, yl, cpu, cpl, mwu, mwl):
        '''
        Set the data of this foil or section.

        ### Note:
        ```text
        xu, yu, xl, yl, cpu, cpl:   ndarray from pyTSFoil
        mwu, mwl:   ndarray from pyTSFoil (do not need built-in Cp2Mw)
        ```
        '''
        cp1u = cpu[-1] + (1-xu[-2])/(xu[-1]-xu[-2])*(cpu[-1]-cpu[-2])
        cp1l = cpl[-1] + (1-xl[-2])/(xl[-1]-xl[-2])*(cpl[-1]-cpl[-2])
        cp1 = 0.5*(cp1u+cp1l)

        self.x  = np.array([1.0]+list(reversed(list(xl[1:]))) + list(xu)+[1.0])
        self.y  = np.array([0.0]+list(reversed(list(yl[1:]))) + list(yu)+[0.0])
        self.Cp = np.array([cp1]+list(reversed(list(cpl[1:]))) + list(cpu)+[cp1])
        
        '''
        mw1u = mwu[-1] + (1-xu[-2])/(xu[-1]-xu[-2])*(mwu[-1]-mwu[-2])
        mw1l = mwl[-1] + (1-xl[-2])/(xl[-1]-xl[-2])*(mwl[-1]-mwl[-2])
        mw1 = 0.5*(mw1u+mw1l)
        self.Mw = np.array([mw1]+list(reversed(list(mwl[1:]))) + list(mwu)+[mw1])
        '''
        self.Mw = self.Cp2Mw()

        iLE = np.argmin(self.x)

        fmw = interp1d(self.x[iLE:], self.Mw[iLE:], kind='cubic')
        gu  = interp1d(self.x[iLE:], self.y [iLE:], kind='cubic')
        x_  = np.append(self.x[iLE:0:-1], self.x[0])
        y_  = np.append(self.y[iLE:0:-1], self.y[0])
        gl  = interp1d(x_, y_, kind='cubic')

        self.xx = np.arange(0.0, 1.0, 0.001)
        self.yu = gu(self.xx)
        self.yl = gl(self.xx)
        self.mu = fmw(self.xx)

        self.iLE = iLE

    def extract_features(self):
        '''
        Extract flow features list in the dictionart.
        '''
        self.locate_basic()
        self.locate_geo()
        self.locate_shock()
        self.aux_features()

    def output_features(self, fname="feature2d.txt", append=True):
        '''
        Output all features to file.

        ### Output order:
        ```text
        feature:    keys of feature dictionary
        key:        'X', 'Mw', 'Cp'
        ```
        '''

        keys = ['X','Mw','Cp']
        features = ['L', 'T', 'Q', 'M', 'F', '1', 'U', '3',
                    'Cu', 'Cl', 'tu', 'tl', 'tm', 'L13',
                    'lSW', 'DCp', 'Err', 'CLU', 'kaf']

        if not os.path.exists(fname):
            append = False

        if not append:
            f = open(fname, 'w')
        else:
            f = open(fname, 'a')

        for feature in self.xf_dict.keys():
            
            if len(self.xf_dict[feature])==2:
                value = self.getValue(feature)
                f.write('%10s   %15.6f \n'%(feature, value))
                continue

            for key in keys:
                name = key+'-'+feature
                value = self.getValue(feature, key)
                f.write('%10s   %15.6f \n'%(name, value))

        f.close()


#* ========================================
#* Supportive functions
#* ========================================

def ratio_vec(x0, x1, x):
    '''
    Calculate distance s to vector x1-x0.

    ### Inputs:
    ```text
    x0, x1: ndarray, start and end point of the vector
    x:      ndarray, current point
    ```

    ### Return:
    ```text
    s:  distance to line
    t:  ratio of (projected |x0x|) / |x0x1|
    ```
    '''
    l0 = np.linalg.norm(x0-x1) + 1e-20
    l1 = np.linalg.norm(x0-x ) + 1e-20
    v  = (x1-x0)/l0
    l2 = np.dot(v, x-x0)
    t  = l2/l1
    s  = np.sqrt(l1**2 - l2**2)

    return s, t

def curve_curvature(x, y):
    '''
    Calculate curvature of points in the curve

    ### Inputs:
    ```text
    x, y: points of curve (list or ndarray)
    ```

    ### Return:
    ```text
    curv: ndarray
    ```
    '''
    nn = len(x)
    if nn<3:
        raise Exception('curvature needs at least 3 points')
    
    curv = np.zeros(nn)
    for i in range(1, nn-1):
        X1 = np.array([x[i-1], y[i-1]])
        X2 = np.array([x[i  ], y[i  ]])
        X3 = np.array([x[i+1], y[i+1]])

        a = np.linalg.norm(X1-X2)
        b = np.linalg.norm(X2-X3)
        c = np.linalg.norm(X3-X1)
        p = 0.5*(a+b+c)
        t = p*(p-a)*(p-b)*(p-c)
        R = a*b*c
        if R <= 1.0E-12:
            curv_ = 0.0
        else:
            curv_ = 4.0*np.sqrt(t)/R

        a1 = X2[0] - X1[0]
        a2 = X2[1] - X1[1]
        b1 = X3[0] - X1[0]
        b2 = X3[1] - X1[1]
        if a1*b2 < a2*b1:
            curv_ = -curv_

        curv[i] = curv_

    curv[0] = curv[1]
    curv[-1] = curv[-2]

    return curv
