

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


def cos(theta):
    return np.cos(theta * np.pi / 180)

def sin(theta):
    return np.sin(theta * np.pi / 180)
