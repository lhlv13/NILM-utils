# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 17:04:20 2022

@author: Yuyi
"""
import numpy as np


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