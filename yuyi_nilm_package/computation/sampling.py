# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:09:27 2022

@author: Yuyi
"""
import numpy as np
from .preprocessing import *

def downSamplingByZerosCrossing(wave, new_sampling_points_of_T, zeros=None):
    """ 降採樣 
    參數
    --------
    wave: 要降採樣的波型
    new_sampling_points_of_T: 更改後的每周期取樣點數 ex: 500 變 32 ， new_sampling_points_of_T=32 
    
    
    return
    --------
    降採樣後的波型
    """
    
    
    if zeros is None:
        zeros = zeroCrossing(wave)
    len_zeros = len(zeros)
    if len(zeros) == 1:
        new_wave = downSamplingFromFullWaves(wave, new_sampling_points=new_sampling_points_of_T)
    
    else:
        new_wave = []
        for i in range(len_zeros):
            if i == (len_zeros-1):
                start, end = zeros[i], None
            else:
                start, end = zeros[i], zeros[i+1]
            temp_wave = downSamplingFromFullWaves(wave[start:end], new_sampling_points=new_sampling_points_of_T)
            for value in temp_wave:
                new_wave.append(value)
    
    return np.array(new_wave)


def downSamplingFromFullWaves(wave, new_sampling_points):
    new_wave = []
    size = len(wave)
    stride = (size-1) / (new_sampling_points-1)
    index = 0
    while index < size:
        new_wave.append(wave[int(index)])
        index += stride
        
    ## 假如不幸採樣點數多於預計採樣點數 就刪掉後面的吧
    if len(new_wave) > new_sampling_points:
        times = len(new_wave) - new_sampling_points
        for i in range(times):
            new_wave.pop()
    
    return np.array(new_wave)
    