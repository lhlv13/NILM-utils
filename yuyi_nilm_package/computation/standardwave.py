# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:13:55 2022

@author: Yuyi
"""
import numpy as np
import matplotlib.pyplot as plt 

def cos_wave(A, frequency, sample_points_of_T, phi, t):
    """ 
    Input
    ----------
    A: 振幅
    frequency: 頻率
    sample_points_of_T: 每周期取樣點數 
    phi: 相位
    t: 時間(s)
    """
    sample_frequency = frequency * sample_points_of_T
    Ts = 1 / sample_frequency
    n = t / Ts
    n = np.arange(n)
    cos = A * np.cos(2*np.pi*frequency*n*Ts + phi*(np.pi/180)) 
    return n, cos   ## plt.plot(n, cos)

def sin_wave(A, frequency, sample_points_of_T, phi, t):
    sample_frequency = frequency * sample_points_of_T
    Ts = 1 / sample_frequency
    n = t / Ts
    n = np.arange(n)
    sin = A * np.sin(2*np.pi*frequency*n*Ts + phi*(np.pi/180)) 
    return n, sin   ## plt.plot(n, sin)