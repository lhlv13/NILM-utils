# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:44:07 2022

@author: Yuyi

"""

""" 使用方式
    
    import yuyi_nilm_package as yuyi
    import torch, torchvision
    from torch import nn
    import torch.nn.functional as F
    import torchvision.models as models
    
    
    def main():
        train_dataset = yuyi.classification.Classify_Dataset("train_img/train")
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=32,
                                                    shuffle=True,
                                                    drop_last=False)
        test_dataset = yuyi.classification.Classify_Dataset("train_img/val")
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                    batch_size=len(test_dataset),
                                                    shuffle=True,
                                                    drop_last=False)
        
        ## {標籤 : 數字}
        label_dict = train_dataset.getLabeldict()
        
        
        
        
        ## 模型
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, 7, stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        
        ## 訓練
        epochs = 20
        lr = 0.00001
        opt = "adam"
        classifyObj = yuyi.classification.Classify(label_dict)
        classifyObj.train(train_loader, test_loader, model, epochs=epochs, lr=lr, opt=opt)
        
        ## 評估
        classifyObj.evaluate(test_loader)
"""


from .classiifier import *
from torch.utils.data import Dataset
import os
from PIL import Image 
import sklearn.metrics as metrics
import seaborn as sns

class Classify_Dataset(Dataset):
    """ 
    torch 的 dataset處理
    """
    def __init__(self, img_folder, input_img_size=(64, 64), transforms=None):
        """
        Input:
        ----------
        img_folder: 路徑資料夾內放有許多以標籤名稱命名的資料夾，資料夾放每一類的 images
        input_img_size: 輸入圖片的大小，需與自己建構的模型一樣
        transforms: 自定義的 transforms
        """
        self.label_name = {}
        label_num = 0
        if not os.path.isdir(img_folder):
            raise FileNotFoundError("path is error!!!")
        self.x = []
        self.y = []
        img_list = []
        img_list = readImgFolder(img_folder)
        for img_path, class_name in img_list:
            img = Image.open(img_path)
            if transforms:
                img = transforms(img)
            else:
                img = img.resize((input_img_size[0], input_img_size[1]))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            ## 將label 改成數字
            if class_name not in self.label_name:
                self.label_name[class_name] = label_num
                label_num += 1
            
            class_name = self.label_name[class_name]
            self.x.append(img)
            self.y.append(class_name)
        self.x = torch.Tensor(self.x)
        # self.y = torch.Tensor(self.y)
        self.len = len(self.x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
    def getLabeldict(self):
        return self.label_name



#####################################################################
class Classify():
    """ 包含訓練，評估模型 ， 使用前要先 dataloader """
    def __init__(self, label_dict):
        """  
        Input:
        ----------
        label_dict: 可從 Classify_Dataset 類別 取得， 結構為 ==> {標籤名 : 對應數字}
        """
        self.__model = None
        self.__label_dict = label_dict  ## {標籤n : 數字n}
        self.__label_dict_reverse = {}
        for key, value in self.__label_dict.items():
            self.__label_dict_reverse[value] = key
        
    def train(self, train_loader, test_loader, model, epochs=10, lr=0.0001, opt="sgd", early_stop=200):
        """ 訓練模型 """
        self.__model = train(train_loader, test_loader, model, epochs, lr, opt, early_stop)
        return self.__model 
    
    def evaluate(self, test_loader):
        """ 評估模型 """
        device = "cpu"
        self.__model.to(device)
        self.__model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = self.__model(x)
                
        pred = pred[:].argmax(1)
        out1 = metrics.classification_report(y, pred, output_dict=True)
        print("Accuracy:  ",out1["accuracy"])
        print(out1)
        
        ## 畫圖
        label_names = [key for key in self.__label_dict]
            
        
        sns.set()
        y_true = []
        for y_unit in y:
            y_true.append(self.__label_dict_reverse[int(y_unit)])
        y_pred = []
        for y_unit in pred:
            y_pred.append(self.__label_dict_reverse[int(y_unit)])
        cm = metrics.confusion_matrix(y_true, y_pred, labels=label_names)
        # cm = metrics.confusion_matrix(y_true, y_pred)
        print(cm)
        sns.heatmap(cm, annot=True, cmap="Pastel1", fmt=".20g", xticklabels=label_names, yticklabels=label_names)
        # sns.heatmap(cm, annot=True, cmap="Pastel1", fmt=".20g")
        
        
        
    
    





