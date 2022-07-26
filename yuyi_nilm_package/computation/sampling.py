# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:09:27 2022

@author: Yuyi
"""
import numpy as np
from .preprocessing import *

def useDownsamplingByZerocrossing(wave, new_sampling_points):
    """ 降採樣 
    參數
    --------
    wave: 要降採樣的波型
    new_sampling_points: 更改後的每周期取樣點數 ex: 500 變 32 ， new_sampling_points=32 
    
    
    return
    --------
    降採樣後的波型
    """
    zero = zeroCrossing(wave)
    new_wave = []
    for i in range(len(zero)-1):
        l = zero[i+1] - zero[i] -1  
        index_unit = l / (new_sampling_points - 1)
        index = zero[i]
        for j in range(new_sampling_points):
            new_wave.append(wave[int(index)])
            index += index_unit
        
    return np.array(new_wave)


# def useDownsamplingByAllWaves(wave, stride):
    
    