import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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

class Series():
    '''
    Operations of series of aero-parameters.i.e. AoA, Cl etc., often 
    calculated from a series of CFD whose Cl are set in a range.

    Currently this class is used for buffet analyse, thus datas
    of AoA and Cl is compulsive. 

    param:
    ---
    `index`     the index of the series\n
    `AoA`       series of AoA, in degree, in a list
        the length of the series is set by the length of AoA
    `CL`        series of CL
    `**kwargs`  other series, given by format:
        parameter name = [series data]

    props:
    ---
    `buffet_cl` buffet cl



    '''
    def __init__(self, index, AoA, CL, **kwargs):
        self.index = index
        self.seriesLength = 0
        self.seriesData = {}
        self.linearSection = (0, 0, 0, 0)

        self._buffet_cl = 0.0
        self._buffet_cl_flag = False
        self.buffet_cl_precise = 0.001
        
        if isinstance(AoA, list) and isinstance(CL, list) and len(AoA) == len(CL):
            self.seriesData['AoA'] = np.array(AoA)
            self.seriesData['CL'] = np.array(CL)
            self.seriesLength = len(AoA)
        else:
            raise Exception("series AoA or CL value is not a list, or their size is not equal")

        for key in kwargs:
            if isinstance(kwargs[key], list):
                if len(kwargs[key]) == self.seriesLength:
                    self.seriesData[key] = np.array(kwargs[key])
                else:
                    print(len(self.seriesData['AoA']), "series %s value size not equal" % key, [len(self.seriesData[key]) for key in self.seriesData])
            else:
                print("series %s value is not a list" % key)
        # print(f"A series with keys: {tuple(self.seriesData.keys())} has been created")
    
    @property
    def buffet_cl(self):
        if self._buffet_cl_flag:
            return self._buffet_cl
        else:
            print(f"series No. {self.index}'s buffet cl not cal")
            return 0.0
    
    def find_linear_section(self, stMethod='single-shock', edMethod='R2', xkey='AoA', ykey='CL'):
        '''
        find the linear section of the curve xkey-ykey
        currently only developed for linear section of
        supercritial airfoil, whose start point decided
        by single shock and end point by linear regression

        paras:
        ---
        `stMethod`  method to judge start point of lin. sec.
            `single-shock`(def)     first point without single shock, judged by first X1 > 0 point
        `edMethod`  method to judge end point of lin. sec.
            `R2`(def)               linear regression from st. point, R2 < 0.998
        `xkey`      input para's key in series
        `ykey`      output para's key in series

        update:
        ---
        `self.linearSection`
            (start point index, end point index, slope = coef_, intercept)

        return:
        ---
        `history`   (xhistory, score, dYdX)
        `xhistory`  xkey's value of point which have been evaluated
        `score`     lin. reg. 's score of each point been evaluated
        `dYdX`      lin. reg. 's coef  of each point been evaluated

        '''
        CL = self.seriesData[ykey]
        AoA = self.seriesData[xkey] # / 180 * 3.14
        X1 = self.seriesData['X1']
        startIDX = 0
        while not (X1[startIDX] > 0 and X1[startIDX + 1] > 0):
            startIDX += 1

        xhistory = []
        score = []
        dYdX = []
        for endIDX in range(startIDX + 2, len(self.seriesData[xkey])):
            
            AoAX = AoA[startIDX:endIDX].reshape(-1, 1)
            CLY = CL[startIDX:endIDX]

            reg = LinearRegression().fit(AoAX, CLY)
            xhistory.append(AoA[endIDX])
            score.append(reg.score(AoAX, CLY))
            dYdX.append(reg.coef_)

            if reg.score(AoAX, CLY) < 0.998:
                break
        else:
            print("can find endIDX")

        AoL0 = -reg.intercept_ / reg.coef_

        self.linearSection = (startIDX, endIDX - 1, float(reg.coef_), float(reg.intercept_))
        
        return xhistory, score, dYdX

    def _buffet_crit_sep(self):
        CL = self.seriesData['CL']
        mUy = self.seriesData['mUy']
        CLGrid  = np.arange(CL[0],  CL[-1] - self.buffet_cl_precise, self.buffet_cl_precise)
        
        f_mUy = interp1d(CL, mUy, kind='cubic')
        mUyNew = 1.0
        for i in range(CLGrid.shape[0]):
            CLn   = CLGrid[i]
            mUyOld = mUyNew
            mUyNew  = f_mUy(CLn)
            if mUyNew <= 0.0 and mUyOld > 0.0:
                self._buffet_cl = CLn
                return True
        else:
            return False
    
    def _buffet_crit_slrd(self, cri, delta):
        AoA = self.seriesData['AoA']
        criData = self.seriesData[cri]
        dAoA = self.buffet_cl_precise * 10
        AoAGrid  = np.arange(AoA[0], AoA[-1] - dAoA + 0.5, dAoA)
        if self.linearSection == (0,0,0,0):
            self.find_linear_section()
        coef = LinearRegression().fit(AoA[self.linearSection[0]:self.linearSection[1]].reshape(-1, 1), criData[self.linearSection[0]:self.linearSection[1]]).coef_ - delta
        fCLAoA = Fitting(AoA, criData)
        flag = False
        
        dYdX = []

        for i in range(1, AoAGrid.shape[0] -1):

            pointp1 = fCLAoA(AoAGrid[i + 1])
            point1 = fCLAoA(AoAGrid[i])
            pointm1 = fCLAoA(AoAGrid[i - 1])

            localdCLdA = 0.5 * (pointp1 - pointm1) / dAoA
            
            if localdCLdA < coef and dYdX[-1] > coef:
                if cri == 'CL':
                    self._buffet_cl = point1
                else:
                    fCLA = Fitting(AoA, self.seriesData['CL'])
                    self._buffet_cl = fCLA(AoAGrid[i])
                flag = True

            dYdX.append(localdCLdA)
        
        return flag, dYdX

    def _buffet_crit_crmx(self, cri):
        AoA = self.seriesData['AoA']
        criData = self.seriesData[cri]
        dAoA = self.buffet_cl_precise * 10
        AoAGrid  = np.arange(AoA[0], AoA[-1] + 0.5, dAoA)

        fCLAoA = Fitting(AoA, criData)
        flag = False

        RCL = [-1] # need to compare with RCL[-1]
        for i in range(1, AoAGrid.shape[0] -1):

            pointp1 = fCLAoA(AoAGrid[i + 1])
            point1 = fCLAoA(AoAGrid[i])
            pointm1 = fCLAoA(AoAGrid[i - 1])

            localdCLdA = 0.5 * (pointp1 - pointm1) / dAoA
            locald2CLdA2 = (pointp1 + pointm1 - 2 * point1) / (dAoA)**2
            localRCL = - locald2CLdA2 / (1 + localdCLdA**2)**1.5
            
            if localRCL < RCL[-1] and RCL[-1] > RCL[-2]:
                if cri == 'CL':
                    self._buffet_cl = pointm1
                else:
                    fCLA = Fitting(AoA, self.seriesData['CL'])
                    self._buffet_cl = fCLA(AoAGrid[i - 1])
                flag = True

            RCL.append(localRCL)
        
        RCL.pop(0) # pop the first element, -1
        return flag, RCL
    
    def set_buffet_crit(self, method='sep', cri='CL', delta= 0.1, precise=0.001):
        '''
        set the buffet cl 's critia

        param:
        ---
        `method`    critia
            sep         initial seperation
            slope-des   slope descend delta
            max-curv    max curvature
        `cri`       the parameter used for curve against AoA

        `delta`     the value descend

        `precise`   the interval of cl to analysis

        '''
        self.buffet_cl_precise = precise
        if method == 'sep':
            self._buffet_cl_flag = self._buffet_crit_sep()
        elif method == 'slope-des':
            self._buffet_cl_flag, value = self._buffet_crit_slrd(cri, delta)
        elif method == 'max-curv':
            self._buffet_cl_flag, value = self._buffet_crit_crmx(cri)
        else:
            raise Exception("xx")
        return self._buffet_cl_flag

    def change_cm_pivot(self, fromP, toP, reverse=False):
        '''
        change the pivot of cm(c.w. is +???)

        param:
        ---
        `fromP`     the pivot currently cm use
        `toP`       the pivot you want cm to use
            LE      leading edge
            AC      aerodynamic center
            TE      tailing edge
        `reverse`   whether to reverse current cm
        '''
        dic = {'LE': 0.0, 'AC': 0.25, 'TE': 1.0}
        deltaC = dic[toP] - dic[fromP]
        if reverse:
            self.seriesData['Cm'] = -self.seriesData['Cm']
        self.seriesData['Cm'] = self.seriesData['Cm'] - deltaC * self.seriesData['CL'] * np.cos(self.seriesData['AoA'] / 180 * 3.14)