# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:16:49 2022

@author: Yuyi
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import copy

import yuyi_nilm_package
line1 = []
line2 = []
line3 = []



class Wave():
    def __init__(self, file_path=None):
        """
        Input
        -----
            file_path : dataset 路徑
        
        使用方式
        -------
            waveObj = Wave().read(file_Path)  ## file_path : 波型存放路徑
            waveObj = Wave().init_wave(original_v_wave, original_i_wave)
        """
        self.__file_path = file_path
        
        ## 各種紀錄
        self.__original_v_wave = None
        self.__original_i_wave = None
        self.__rms_v_wave = None
        self.__rms_i_wave = None
        
        ## 物件
        self.__readObj = yuyi_nilm_package.read_write.Read_wave()
        self.hybridObj = yuyi_nilm_package.event_detection.Hybrid_approach()
        self.__drawObj = yuyi_nilm_package.plot.DrawWave()
        
        if file_path:
            if os.path.isfile(self.__file_path):
                self.read(self.__file_path)
            

    ### 讀取檔案或波型
    def read(self, path):
        voltage, current = self.__readObj.auto_read(path).getWaves()
        self.init_wave(voltage, current)
        self.__file_path = path
        return self
    
    def init_wave(self, original_v_wave, original_i_wave):
        """
        Input
        -----
        v_wave: original voltage wave by measuring
        i_wave: original current wave by measuring
        """
        if len(original_v_wave) != len(original_i_wave):
            raise Exception("Two waves must have same length!!")
        self.__original_v_wave = original_v_wave if(type(original_v_wave) != type(np.array([1]))) else np.array(original_v_wave)
        self.__original_i_wave = original_i_wave if(type(original_i_wave) == type(np.array([1]))) else np.array(original_i_wave) 
        self.__zero_crossing = yuyi_nilm_package.computation.zeroCrossing(self.__original_v_wave)
        self.__rms_v_wave = yuyi_nilm_package.computation.rms(self.__original_v_wave)
        self.__rms_i_wave = yuyi_nilm_package.computation.rms(self.__original_i_wave, base_wave=self.__original_v_wave) 
        
        return self
    
    def get_I(self):
        """ 原始電流波型 """
        return copy.deepcopy(self.__original_i_wave)
    def get_V(self):
        """ 原始電壓波型 """
        return copy.deepcopy(self.__original_v_wave)
    def get_Irms(self):
        """ rms 電流波型 """
        return copy.deepcopy(self.__rms_i_wave)
    def get_Vrms(self):
        """ rms 電壓波型 """
        return copy.deepcopy(self.__rms_v_wave)
    def get_zeroCrossing(self):
        """ 過零點 list """
        return copy.deepcopy(self.__zero_crossing)
    def get_power(self):
        """  視在功率 """
        return copy.deepcopy(self.__rms_i_wave * self.__rms_v_wave)
    
    
    
    ### 畫圖
    def plotWave(self, wave, title=None, xlabel="Sample Points", ylabel="Altitude", segment=":"):
        """ 畫單一波型圖  
        Input
        -----------
        segment : 切片，跟list用法一樣，只是放在字串裡面 ex: ":", ":10", "10:20" , "10:" , "10:30:2"
        """
        self.__drawObj.plotWave(wave=wave, title=title, xlabel=xlabel, ylabel=ylabel, segment=segment)
    
    def plotWave_withEvents(self, wave, events,  title=None, xlabel="Sample Points", ylabel="Altitude", isEvent_1=False, segment=":"):
        """ 畫單一波型圖 + 事件點(on、off合併) 
        Input
        -----------
        segment : 切片，跟list用法一樣，只是放在字串裡面 ex: ":", ":10", "10:20" , "10:" , "10:30:2"
        """
        self.__drawObj.plotWave_withEvents(wave=wave, events=events, title=title, xlabel=xlabel, ylabel=ylabel, isEvent_1=isEvent_1, segment=segment)
    
    
        
        
    
        


    


def main():
    dirs = os.listdir("../dataset/")
    for path in dirs:
        route = os.path.join("../dataset", path)
        wave = Wave().read(route)
        # wave.plot_wave(wave.get_power(), title=path, segment=":1000", ylabel="W")
        
        
        
        a = wave.hybridObj.run(wave.get_Irms())
        
        # print(len(wave.get_rms_I()))
        # print(len(a))
        # # plt.plot(a[150:200])
        # plt.plot(a[:],c='r')
        # plt.plot(wave.get_rms_I(),c='yellow')
        # break
    
    
    
    ###---------------------------------------------------
    ## 測試 Read_wave 的所有 方法
    # dirs = os.listdir("./")
    # for d in dirs:
    #     waves = Read_wave(d).getWaves()
    #     print(waves.shape)
        
    ###---------------------------------------------------
        
    
    
    
if __name__ == "__main__":
    main()