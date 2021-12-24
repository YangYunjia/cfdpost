import math

import subprocess
import tempfile
import os

ga = 1.4
R = 287

coef8 = math.sqrt(ga / R * (2.0 / (ga + 1))**((ga + 1) / (ga - 1)))

def std_atomsphere(h, tref=288.15, rref=1.225, pref=101325, muref=1.72e-5):
    '''
    docin.com/p-70811798.html
    杨炳尉 标准大气参数的公式表示 宇航学报 1983年1月

    '''
    if h <= 11:
        w = 1 - h / 44.3308
        t = tref * w
        r = rref * w**4.2559
        p = pref * w**5.2559
    elif h <= 20:
        w = math.exp((14.9647 - h) / 6.3416)
        t = 216.65
        p = pref * 0.11953 * w
        r = rref *  1.5898 * w
    elif h <= 32:
        w = 1 + (h - 24.9021) / 221.552
        t = 221.55
        p = pref * 0.025158 * w**-34.1629
        r = rref * 0.032722 * w**-35.1629
    else:
        raise ValueError("h < 0 or h > 32")
    
    mu = muref * (t / 273.11)**1.5 * 383.67 / (t + 110.56)
    a = 20.0468 * t**0.5

    return {'temperature': t, 'pressure': p, 'density': r, 'viscosity': mu, 'soundspeed': a}

def ideal_mfr(pt8, tt8, A8):
    return coef8 * pt8 / tt8**0.5 * A8

def ideal_thrust(pt7, tt7, p9, m8):
    npr = pt7 / p9
    v9id = math.sqrt(2 * ga * R / (ga-1) * tt7 * (1 - npr**(-(ga-1)/ga)))
    thrustid = m8 * v9id
    return thrustid, v9id
    