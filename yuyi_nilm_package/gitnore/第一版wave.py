import sys
import os
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import copy

from src.read_write.Read_wave import *
from src.event_detection.Hybrid_Approach import *
from src.plot.Draw_wave import *
line1 = []
line2 = []
line3 = []


class Wave():
    def __init__(self, rms_T_num = 1):
        """
        Input
        -----
            self.__rms_T_num : ### 多少個週期計算一筆rms電壓電流資料  假如 60 市電也是60hz 就很像用三用電表一秒取一筆rms值
            
        
        使用方式
        -------
            waveObj = Wave().read(file_Path)  ## file_path : 波型存放路徑
            waveObj = Wave().init_wave(original_v_wave, original_i_wave)
        """
        self.__file_name = None
        self.__rms_T_num = rms_T_num    ### 多少個週期計算一筆rms電壓電流資料  假如 60 市電也是60hz 就很像用三用電表一秒取一筆rms值
        self.__original_v_wave = None
        self.__original_i_wave = None
        self.__len = 0 
        self.__first_sample_point_of_T = None
        self.__T_len = 0
        self.__rms_v = None
        self.__rms_i = None
        self.__apparent_power = None
        ## 物件
        self.__readObj = Read_wave()
        self.hybridObj = Hybrid_approach()
        self.__drawObj = Draw_wave()
            
    
    def get_file_name(self):
        return self.__file_name
    
    def get_V(self):
        return copy.deepcopy(self.__original_v_wave)
    
    def get_I(self):
        return copy.deepcopy(self.__original_i_wave)
    
    def get_rms_V(self):
        return copy.deepcopy(self.__rms_v)
    
    def get_rms_I(self):
        return copy.deepcopy(self.__rms_i)
    
    def get_apparent_power(self):
        return copy.deepcopy(self.__apparent_power)
    
    def get_first_sample_point_of_T(self):
        return copy.deepcopy(self.__first_sample_point_of_T)
    
    
    def event_detection(self, wave, method_name):
        method_dict = {"hybrid_approach" : Hybrid_approach()}
        return method_dict[method_name.lower()].run(wave)
        
    
    ### draw
    def plot_event_detection(self, data, on_list, off_list, title="", scope=None, save=None):
        self.__drawObj.plot_event_detection(data, on_list, off_list, title, scope, save=save)
    
    

    ### 讀取檔案或波型
    def read(self, path):
        waves = self.__readObj.auto_read(path).getWaves()
        self.init_wave(waves[0], waves[1])
        self.__file__name = path
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
        self.__len = len(original_v_wave) 
        self.__first_sample_point_of_T = self.__zero_crossing()
        self.__T_len = len(self.__first_sample_point_of_T)
        
        self.__calculate_rms(self.__rms_T_num)
        self.__calculata_apparent_power()
        return self
    
    def __zero_crossing(self):
        """
        尋找零交越的樣本點 
        """

        first_points = []
        pre_point = 1
        for i in range(self.__len):
            now_point = self.__original_v_wave[i]
            if now_point==0:
                first_points.append(i)
            elif((pre_point<0) and (now_point>0)):
                first_points.append(i)
            pre_point = now_point
        return np.array(first_points)
    
    
    def rms(self, wave_list):
        wave = copy.deepcopy(wave_list)
        if(type(wave) != type(np.array([1]))):
            wave = np.array(wave)
        return np.sqrt((wave**2).sum() / len(wave))
    
    
    def __calculate_rms(self, period_num=1):  ## Period_num: 幾個週期當作一筆rms資料
        rms_v_list = []
        rms_i_list = []
        start_point = None
        end_point = None
      
        for i in range(0, self.__T_len-period_num, period_num):
            start_point = int(self.__first_sample_point_of_T[i])
            end_point = int(self.__first_sample_point_of_T[i+period_num])
            
            rms_v_list.append(self.rms(self.__original_v_wave[start_point:end_point]))
            rms_i_list.append(self.rms(self.__original_i_wave[start_point:end_point]))
        ## 最後一筆
        rms_v_list.append(self.rms(self.__original_v_wave[end_point:]))
        rms_i_list.append(self.rms(self.__original_i_wave[end_point:]))
        self.__rms_v = np.array(rms_v_list)
        self.__rms_i = np.array(rms_i_list)
    
    def __calculata_apparent_power(self):
        apparent_power_list = []
        for i in range(len(self.__rms_v)):
            apparent_power_list.append(self.__rms_v[i] * self.__rms_i[i])

        self.__apparent_power = np.array(apparent_power_list)
    
        
        
    
        


    


def main():
    dirs = os.listdir("../dataset/")
    for path in dirs:
        wave_name = path.split(".")[0]
        wave = Wave().read_file("dataset/"+path)
        a = wave.hybridObj.run(wave.get_rms_I())
        
        print(len(wave.get_rms_I()))
        print(len(a))
        # plt.plot(a[150:200])
        plt.plot(a[:],c='r')
        plt.plot(wave.get_rms_I(),c='yellow')
        break
    
    
    
    ###---------------------------------------------------
    ## 測試 Read_wave 的所有 方法
    # dirs = os.listdir("./")
    # for d in dirs:
    #     waves = Read_wave(d).getWaves()
    #     print(waves.shape)
        
    ###---------------------------------------------------
        
    
    
    
if __name__ == "__main__":
    main()