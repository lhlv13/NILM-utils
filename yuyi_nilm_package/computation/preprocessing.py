# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:11:47 2022

@author: Yuyi
"""
import numpy as np

def zeroCrossing(wave:list, sample_point_of_T=None)->list:
    """
    功能
    --------
        計算過零點
    Input
    --------
        wave : original_wave (ex: wave of voltage)
    Output
    --------
        return : a list of zero sampling point
    """
    zero_crossing_list = []
    size = len(wave)
    zero_threshold = 0
    if sample_point_of_T:
        pre_i = None
        point_num = sample_point_of_T * 0.75
    for i in range(1, size):
        if((wave[i]>zero_threshold) and (wave[i-1]<=zero_threshold)):
            if sample_point_of_T:
                if pre_i and (i - pre_i) < (point_num):
                    continue
                
            zero_crossing_list.append(i)  ## 增加 零交越點
             
            if sample_point_of_T:
                pre_i = i
                i += sample_point_of_T
                    
                    
                    
    
    return np.array(zero_crossing_list)

def fit_ps(voltage, current, sampPerPeriods=32):
    """ 
    Input:
    ---------
    voltage: 電壓 list
    current: 電流 list
    sampPerPeriods=32
    
    Output:  np.array(output_V), np.array(output_I)
    ----------
    
    """
    ### 找過零點 zero_list
    zero_list = zeroCrossing(voltage)
    
    ### 計算電壓零點偏移量 zero_shift
    zero_shift = [-voltage[index] / (voltage[index+1] - voltage[index]) for index in zero_list]
        
    
    ### 插值法分配個週期取樣點
    output_V = []
    output_I = []
    for i in range(len(zero_shift)-1):
        length = (zero_list[i+1] + zero_shift[i+1]) - (zero_list[i]+zero_shift[i]) ## 兩個修正後的過零點間取樣點數目
        dis = length / sampPerPeriods ## 每個新取樣點間的時間
        for j in range(0, sampPerPeriods):
            k1 = zero_list[i] + zero_shift[i] + dis * (j)
            k2 = int(k1 // 1)
            k3 = k2 + 1
            output_V.append(voltage[k2] + (voltage[k3] - voltage[k2]) * zero_shift[i])
            output_I.append(current[k2] + (current[k3] - current[k2]) * zero_shift[i])
            
    return np.array(output_V), np.array(output_I)

def minMaxScaling(wave):
    if not isinstance(wave, np.ndarray):
        wave = np.array(wave)
    min_v = min(wave)
    max_v = max(wave)
    
    return (wave - min_v) / (max_v - min_v)