# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:28:19 2022

@author: Yuyi
"""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import *





def rms(wave:list, base_wave:list=None)->list:
    """
    功能
    --------
        計算rms值
    Input
    --------
        wave : original_wave (ex: wave of voltage) 
        base_wave : 作為評量過零點的波形，通常是電壓波行
    Output
    --------
        return : a list of rms wave
    """
    rms_list = []
    origin_wave = np.array(wave)
    if base_wave is None:
        base_wave = wave
    zero_crossing_list = zeroCrossing(base_wave)
    for i in range(1, len(zero_crossing_list)):
        start, end = zero_crossing_list[i-1], zero_crossing_list[i]
        rms_value = np.mean(origin_wave[start:end]**2) 
        rms_value = np.power(rms_value, 0.5)
        rms_list.append(rms_value)
    return np.array(rms_list)  ## 長度會比 zeroCrossing 少 1  

def rmsBySetSamplePointNum(wave:list, sample_point_num:int)->list:
    """
    功能
    --------
        計算自定義幾個樣本點數量為一週期的 rms值 
    Input
    --------
        wave : original_wave (ex: wave of voltage) 
        sample_point_num : 多少個取樣點計算一筆 rms值 
        
    Output
    --------
        return : a list of rms wave
    """
    rms_list = []
    origin_wave = np.array(wave)
    size = len(origin_wave)
    assert size > sample_point_num, "sample_point_num大於波型取樣點數!! ==> rmsBySetSamplePointNum()"
    start, end = 0, sample_point_num
    while end < size:
        rms_value = np.mean(origin_wave[start:end]**2)
        rms_value = np.power(rms_value, 0.5)
        rms_list.append(rms_value)
        start, end = end, end+sample_point_num
    return np.array(rms_list)  


def meanOfPeriod(wave:list, base_wave:list=None)->list:
    """
    功能
    --------
        計算每個周期的 mean值， 週期是用過零點判斷
    Input
    --------
        wave : original_wave (ex: wave of voltage) 
        base_wave : 作為評量過零點的波形，通常是電壓波行
    Output
    --------
        return : a list of mean wave
    """
    mean_list = []
    origin_wave = np.array(wave)
    if base_wave is None:
        base_wave = wave
    zero_crossing_list = zeroCrossing(base_wave)
    for i in range(1, len(zero_crossing_list)):
        start, end = zero_crossing_list[i-1], zero_crossing_list[i]
        mean_value = np.mean(origin_wave[start:end])
        mean_list.append(mean_value)
    return np.array(mean_list)  ## 長度會比 zeroCrossing 少 1  


def meanBySetSamplePointNum(wave:list, sample_point_num:int)->list:
    """
    功能
    --------
        計算自定義幾個樣本點數量為一週期的 mean值 
    Input
    --------
        wave : original_wave (ex: wave of voltage) 
        sample_point_num : 多少個取樣點計算一筆 mean值 
        
    Output
    --------
        return : a list of mean wave
    """
    
    mean_list = []
    origin_wave = np.array(wave)
    size = len(origin_wave)
    assert size > sample_point_num, "sample_point_num大於波型取樣點數!! ==> meanBySetSamplePointNum()"
    start, end = 0, sample_point_num
    while end < size:
        mean_value = np.mean(origin_wave[start:end])
        mean_list.append(mean_value)
        start, end = end, end+sample_point_num
    return np.array(mean_list)  











def vi_plot(v_per_T, i_per_T, is_normalize=True, save_img=None):
    """畫 V-I 軌跡圖
    Input:
    ----------
    v_per_T: (list), 一個週期的 V     ## 還要取下個周期的第一個點，這樣軌跡才會連起來
    i_per_T: (list), 一個周期的 I     ## 還要取下個周期的第一個點，這樣軌跡才會連起來
    is_normalize: 平均值正規畫， [-1, 1]
    save_img: path of img to save.
    
    Return:
    ----------
    V, I (也許有均值正規化)
    """
    V, I = list(), list()
    if is_normalize:
        mean_v = np.mean(v_per_T)
        mean_i = np.mean(i_per_T)
        for i in range(len(v_per_T)):
            V.append((v_per_T[i] - mean_v) / (max(v_per_T) - min(v_per_T)))
            I.append((i_per_T[i] - mean_i) / (max(i_per_T) - min(i_per_T)))
    
    else:
        V = v_per_T
        I = i_per_T
    plt.clf()
    plt.figure(figsize=(8,8))
    plt.plot(I, V)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # plt.axis("square")
    if save_img:
        plt.axis("off")
        # width, height = 64, 64
        # plt.rcParams['figure.figsize'] = width, height
        plt.savefig(save_img, dpi=8)
    plt.show()
    return V, I

def plotLinear(img, point_start, point_end):
    r1, c1 = point_start[0], point_start[1]
    r2, c2 = point_end[0], point_end[1]
    ## r1 == r2
    if (r2 == r1):
        start = c1 if c1<c2 else c2
        end = c1 if c1>c2 else c2
        while True:
            if start == end:
                break
            start += 1
            img[r1][start] = 255
        return img
    ## c1 == c2
    if (c2 == c1):
        start = r1 if r1<r2 else r2
        end = r1 if r1>r2 else r2
        while True:
            if start == end:
                break
            start += 1
            img[start][c1] = 255
        return img
    
    ## 
    slope = (c2 - c1) / (r2 - r1)
    if (slope >= -1) and (slope <= 1):
        
        start = r1 if r1<r2 else r2
        end = r1 if r1>r2 else r2
        value = c1 if start==r1 else c2
        while True:
            if start == end:
                break
            ## 更新
            start += 1
            value += slope
            img[start][round(value)] = 255
    
    else:
        slope = 1/ slope
        start = c1 if c1<c2 else c2
        end = c1 if c1>c2 else c2
        value = r1 if start==c1 else r2
        while True:
            if start == end:
                break
            ## 更新
            start += 1
            value += slope
            img[round(value)][start] = 255
    
    
    return img



def vi_plotBinary(v_per_T, i_per_T, img_size=64 , is_link_line = True):
    """ 
    draw vi plot but if your wave is nois, maybe return None.
    """
    V, I = list(), list()
    mean_v = np.mean(v_per_T)
    mean_i = np.mean(i_per_T)
    max_v, min_v = max(v_per_T), min(v_per_T)
    max_i, min_i = max(i_per_T), min(i_per_T)
    if (max_i==min_i):
        return np.zeros((img_size, img_size))
    for i in range(len(v_per_T)):
        V.append((v_per_T[i] - mean_v) / (max_v - min_v))
        I.append((i_per_T[i] - mean_i) / (max_i - min_i))
    output_img = np.zeros((img_size, img_size))
    unit = 2 / img_size  ## -1~1 分成 img_size格
    size = len(V) if len(V)<len(I) else len(I)
    if is_link_line:
        ## 紀錄第一個row, col
        first_row, first_col = int(abs(I[0]-1) / unit), int((V[0]+1) / unit)
        ## 記錄前一個 row,col
        pre_row, pre_col = first_row, first_col
    for i in range(size):
        col = (V[i]+1) / unit   ## 變成0~2 比較好算格子
        row = abs(I[i]-1) / unit
        try:
            col = 0 if col==0 else int(col)-1
            row = 0 if row==0 else int(row)-1
        except:
            # print(V[i], I[i])
            continue
        
        output_img[row][col] = 255
        if is_link_line:
            output_img = plotLinear(output_img, point_start=(pre_row, pre_col), point_end=(row, col))
            pre_row, pre_col = row, col 
        
    
    if is_link_line:
        output_img = plotLinear(output_img, point_start=(pre_row, pre_col), point_end=(first_row, first_col))
    return output_img


        
    


def envelop(wave, zeros:list=None):
    """ 
    Input:
    ----------
        wave: 通常是電流波型
        zeros: 過零點的 list
    
    Output:
    ----------
        (每週期波的最大值位置索引, 每個周期波的最大值), (每週期波的最小值位置索引, 每個周期波的最小值)
    """
    if not type(zeros):
        zeros = zeroCrossing(wave)
    
    envelop_up, envelop_down = list(), list()
    envelop_up_index, envelop_down_index = list(), list()
    
    pre_max, pre_min = None, None
    for i in range(len(zeros)-1):
        start, end = zeros[i], zeros[i+1]
        max_value = max(wave[start : end])
        max_index = np.argmax(wave[start : end]) + zeros[i]
        min_value = min(wave[start : end])
        min_index = np.argmin(wave[start : end]) + zeros[i]
        
        
        envelop_up.append(max_value)
        envelop_up_index.append(max_index)
        envelop_down.append(min_value)
        envelop_down_index.append(min_index)
        
        pre_max, pre_min = max_value, min_value
        
    ## 處理一些錯誤值
    size = len(envelop_up)
    for i in range(1, size-1):
        if (envelop_up[i]<envelop_up[i-1]) and (envelop_up[i]<envelop_up[i+1]):
            envelop_up[i] = (envelop_up[i-1] + envelop_up[i+1]) / 2
        if (envelop_down[i]>envelop_down[i-1]) and (envelop_down[i]>envelop_down[i+1]):
            envelop_down[i] = (envelop_down[i-1] + envelop_down[i+1]) / 2
            
            
    return (np.array(envelop_up_index), np.array(envelop_up)), (np.array(envelop_down_index), np.array(envelop_down))
    
    
    










def main():
    
    x, wave = cos_wave(A=1, frequency=60, sample_frequency=1920, sample_period=1)
    
    zero_list = zeroCrossing(wave)
    rms_list = rms(wave)
    rms_set_sp = rmsBySetSamplePointNum(wave, sample_point_num=32)
    mean_list = meanOfPeriod(wave)
    mean_set_sp = meanBySetSamplePointNum(wave, sample_point_num=32)
    
    
    plt.plot(mean_set_sp)
    plt.ylim((0,2))
    plt.show()
    
    
if __name__ == "__main__":
    main()
            
        