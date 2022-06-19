import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import copy
from math import ceil
from scipy import linalg
import math


def convolution(data, kernel, stride=1):
    data_size = len(data)
    kernel_size = len(kernel)
    value = 0
    conv_list = []
    for i in range(0, data_size-kernel_size+1, stride):
        value = 0
        for j in range(kernel_size):
            value += data[i+j] * kernel[j]
        conv_list.append(value)
    
    return np.array(conv_list)


def convolution_func(wave, func, kernel_size=2, stride=1):
    """ 
    通用的捲積
    parameters
    ---------------
    wave: 1-D array
    func: calculation of convolution you need ==> def func(1-D_list)->value
    assign: 賦值給 kernel 的某一個索引， 例如: kenel=3 ， 你可以選擇 0, 1, 2 
    kernel_size: kernel 大小
    stride: 每次走的步伐
    """
    conv_list = []
    for i in range(0, len(wave)-kernel_size+1, stride):
        values = func(wave[i:i+kernel_size])
        if type(values) == type(np.array([1])):
            for value in values:
                conv_list.append(value)
        else:
            conv_list.append(values)
    
    return np.array(conv_list)
###-----------------------------------------------------------
class Hybrid_approach():
    def __init__(self):
        self.__input_wave = []
        self.__output_wave = []
        
        self.__base_algorithm_list = []
        self.__derivative_list = []
        self.savizky_list = []
        
    
    def run(self, input_wave):
        self.__input_wave = copy.deepcopy(input_wave)
        base_algorithm_list = self.__base_algorithm(kernel_size=18, p_thes=1, T_thes=12)
        derivative_list = self.__derivative_loess(kernel_size=5, d_thres=0.5, t_thres=120)
        savizky_list = self.__savizky_golay(kernel_size=5, p_thres=5)
        
        output_list = copy.deepcopy(savizky_list)
        for i, value in enumerate(output_list):
            if value > 0:
                output_list[i] = 1
            elif value < 0:
                output_list[i] = -1
        
        return output_list
    
    def __base_algorithm(self, kernel_size=18, p_thes=2, T_thes=12):
        """
        原作者設定
        ---------
        n = 0.3s  ## kernel size = 60Hz* 0.3 = 18
        p_th = ?
        T_th = 0.2s
        """
        ## mean
        mean_list = []  ## mean_after - mean_before
        
        input_waves = copy.deepcopy(self.__input_wave)
        
        kernel = [float(1) for i in range(kernel_size)]
        conv_list = convolution(input_waves, kernel) / kernel_size
        # print(len(conv_list))
        # print(len(self.waveobj.get_rms_I()))
        for i in range(len(conv_list)-kernel_size-1):
            mean_list.append(conv_list[i+kernel_size+1]-conv_list[i])
        
        ## P theshold
        p_list = copy.deepcopy(mean_list)
        for i, value in enumerate(p_list):
            if((value < p_thes) and (value > -p_thes)):
                p_list[i] = 0
            
        ## T theshold
        t_list= copy.deepcopy(p_list)
        t_size = len(t_list)
        for i, value in enumerate(t_list):
            if value != 0:
                for j in range(i+1, i+1+T_thes):
                    if(j == t_size):
                        break
                    if value > 0:
                        t_list[j] = 0 if t_list[j]>0 else t_list[j]
                    
                    elif value < 0:
                        t_list[j] = 0 if t_list[j]<0 else t_list[j]
            
        
        ## 紀錄
        repair_size = len(input_waves) - len(t_list)
        repair_list = [0 for i in range(repair_size)]
        output_list = np.concatenate((repair_list[:repair_size//2], t_list, repair_list[repair_size//2:]), axis=0)
        
        self.__base_algorithm_list = np.array(copy.deepcopy(output_list))
        self.__output_wave = copy.deepcopy(self.__base_algorithm_list)
        
        return copy.deepcopy(self.__base_algorithm_list)
        # return mean_list
    
    def __derivative_loess(self, kernel_size=5, d_thres=0.5, t_thres=120):
        """
        原作者設定
        ---------
        d: 0.5  ## 一階導數的閥值只要小於，都當作noise
        T_th: 2s  ## 2s 以內的事件都當作同一事件
        """
        if kernel_size%2!=1:
            raise Exception("kernel_size is odd number")
        
        input_waves = copy.deepcopy(self.__base_algorithm_list)
        
        kernel = [i for i in range(-(kernel_size//2), (kernel_size//2)+1)]
        conv_list = convolution(input_waves, kernel)
        
        ### loess 平滑  時間太久啦!!
        loess_list = copy.deepcopy(conv_list)
        # loess_list = convolution_func(loess_list[:], self.__lowess, kernel_size=20)
        
        ## 導數閥值
        deri_list = copy.deepcopy(loess_list)
        for i in range(len(deri_list)):
            if deri_list[i] < d_thres and deri_list[i]>(-d_thres):
                deri_list[i] = 0
        
        
        ## 時間閥值
        t_list = copy.deepcopy(deri_list)
        t_size = len(t_list)
        for i, value in enumerate(t_list):
            if (i+t_thres) > t_size:
                break
            only_one = 0
            if value > 0:
                max_value = max(t_list[i:i+t_thres])
                for j in range(i, i+t_thres):
                    if t_list[j] == max_value and only_one==0:
                        only_one = 1
                    else:
                        t_list[j] = 0
            
            elif value < 0:
                min_value = min(t_list[i:i+t_thres])
                for j in range(i, i+t_thres):
                    if t_list[j] == min_value and only_one==0:
                        only_one = 1
                    
                    else:
                        t_list[j] = 0
                        
        
        ## 紀錄
        repair_size = len(input_waves) - len(t_list)
        repair_list = [0 for i in range(repair_size)]
        output_list = np.concatenate((repair_list[:repair_size//2], t_list, repair_list[repair_size//2:]), axis=0)
        self.__derivative_list = np.array(copy.deepcopy(output_list))
        self.__output_wave = copy.deepcopy(self.__derivative_list)
        
        # return conv_list
        return copy.deepcopy(self.__derivative_list)
            
        
                
        
    
    
    
    def __savizky_golay(self, kernel_size=5, p_thres=4):
        input_waves = copy.deepcopy(self.__input_wave)
        # input_waves = copy.deepcopy(self.__derivative_list)
        conv_list = convolution_func(input_waves, self.__least_square, kernel_size=kernel_size, stride=kernel_size)
        for i, value in enumerate(conv_list):
            if value < p_thres:
                conv_list[i] = 0
        medium_list = convolution_func(conv_list, self.__medium, kernel_size= 3, stride=1)
        
        repair_size = len(input_waves)-len(medium_list)
        repair_list = [0 for i in range(repair_size)]
        medium_list = np.concatenate((repair_list[:repair_size//2], medium_list, repair_list[repair_size//2 :]), axis=0)
        
        ## 修正基本演算法
        error_range = 5
        base_list = copy.deepcopy(self.__derivative_list)
        for i, value in enumerate(medium_list):
            if value > 0:
                for j in range(i-error_range, i+error_range+1):
                    if base_list[j] > 0:
                        base_list[j] = 0
                
                
        
        return base_list
        
    
    
    def __medium(self, data):
        sorted_data = sorted(data)
        medium_index = (len(data) + 1) // 2
        if data[medium_index] != 0:
            return 0
        
        return sorted_data[medium_index]
    
    
    
    def __least_square(self, data):
        ones = np.ones(len(data))
        line = np.array([i for i in range(len(data))])
        x = np.vstack((ones, line)).T 
        y = data 
        ## 最小平方法
        least = (np.dot(x.T, x))
        least = np.linalg.inv(least)
        least = np.dot(least, x.T)
        least = np.dot(least, y)
        
        output = line * least[1] + least[0]
        
        ## 過濾超出太多的質
        max_v = max(data)
        min_v = min(data)
        for i in range(len(output)):
            if output[i] > max_v:
                output[i] = max_v
            elif output[i] < min_v:
                output[i] = min_v
        return output
    
        
    def __lowess(self, y, f=2./3., iter=1):
        """lowess(x, y, f=2./3., iter=3) -> yest
        Lowess smoother: Robust locally weighted regression.
        The lowess function fits a nonparametric regression curve to a scatterplot.
        The arrays x and y contain an equal number of elements; each pair
        (x[i], y[i]) defines a data point in the scatterplot. The function returns
        the estimated (smooth) values of y.
        The smoothing span is given by f. A larger value for f will result in a
        smoother curve. The number of robustifying iterations is given by iter. The
        function will run faster with a smaller number of iterations.
        """
        x = np.array([i for i in range(len(y))])
        n = len(x)
        r = int(ceil(f * n))
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n)
        for iteration in range(iter):
            for i in range(n):
                weights = delta * w[:, i]
                b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
                A = np.array([[np.sum(weights), np.sum(weights * x)],
                              [np.sum(weights * x), np.sum(weights * x * x)]])
                beta = linalg.solve(A, b)
                yest[i] = beta[0] + beta[1] * x[i]
    
            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2
    
        return yest
 
    
    def __mean(self, kernel)->float:
        """ 
        window: 1-D list
        """
        return sum(kernel) / len(kernel)
    
        