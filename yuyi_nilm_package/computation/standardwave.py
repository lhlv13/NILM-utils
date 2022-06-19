# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:13:55 2022

@author: Yuyi
"""
import numpy as np
import matplotlib.pyplot as plt 

def cos_wave(A, frequency, sample_frequency_point, sample_period, isShow=True):
    """ 
    Input:
    ----------
    A: 振幅
    frequency: 每秒幾個波
    sample_frequency_point:每秒取多少點
    sample_period: 取多少秒
    """
    x = np.linspace(0, sample_period, sample_frequency_point)
    y = A * np.cos(2*np.pi* frequency* x)
    if isShow:
        plt.plot(x, y)
        plt.show()
    return x, y

def sin_wave(A, frequency, sample_frequency, sample_period, isShow=True):
    x = np.linspace(0, sample_period, sample_frequency)
    y = A * np.sin(2*np.pi* frequency* x)
    if isShow:
        plt.plot(x, y)
        plt.show()
    return x, y