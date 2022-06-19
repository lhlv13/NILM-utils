import os
import copy
import numpy as np
import scipy.io 

class Read_wave():
    """
    Method
    ------
    get_file_name(): 取得讀取檔案的名稱(路徑) 
    auto_read(path): 自動選擇以下方法讀取  return self
    read_csv(path): 讀取 csv檔  return 波型: [[電壓], [電流]]
    read_mat(path): 讀取 mat檔  return 波型: [[電壓], [電流]]
    read_txt(path): 讀取 txt檔  return 波型: [[電壓], [電流]]
    getWaves(): 取得讀取檔案後，得到的波型 [[電壓], [電流]]
    """
    def __init__(self, file_path=None):
        self.__waves = None
        
        self.__file_path = file_path
        if file_path:
            if not os.path.isfile(file_path):
                raise FileNotFoundError("No this file or path!!")
            self.auto_read(self.__file_path)
            
            
    
    def auto_read(self, path):
        self.__file_path = path
        extention = self.__file_path.split(".")[-1].lower()
        if extention == "csv":
            waves = self.read_csv(self.__file_path)
        elif extention == "mat":
            waves = self.read_mat(self.__file_path)
        elif extention == "txt":
            waves = self.read_txt(self.__file_path)
        else:
            raise Exception("This method can't deal ",self.__file_path)
        return self
            
    def get_file_name(self):
        return self.__file_path
    
    def getWaves(self):
        return copy.deepcopy(self.__waves)
    
    def read_csv(self, path)->list:
        waves_list = []
        with open(path, "r") as f:
            datas = f.readlines()[9:]   ## 去掉 前8行 不重要的資訊
            for i in range(len(datas)):
                line_datas = datas[i].strip("\n").strip().split(",")[1:]
                waves_list.append([float(line_datas[0]), float(line_datas[1])])
            self.__file_path = path
            self.__waves = np.array(waves_list).T
            return  self.__waves
    
    def read_mat(self, path)->list:
        # f = h5py.File(path, 'r')
        mat = scipy.io.loadmat(path)
        for key, value in mat.items():
            if (type(value)==type(np.array([1]))): ## 這行在找 array 
                 self.__file_path = path
                 self.__waves = value.T
                 return  self.__waves
        return None
    
    def read_txt(self, path)->list:
        waves_list = []
        with open(path, "r") as f:
            datas = f.readlines()[10:]   ## 去掉 前9行 不重要的資訊
            for i in range(len(datas)):
                line_datas = datas[i].strip("\n").strip().split(",")[1:]
                waves_list.append([float(line_datas[0]), float(line_datas[1])])
            self.__file_path = path
            self.__waves = np.array(waves_list).T
            return self.__waves