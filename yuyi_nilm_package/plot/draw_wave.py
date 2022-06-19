import matplotlib.pyplot as plt
import numpy as np


class DrawWave():
    
    def __segment(self, segment, wave_length):
        """ 處理切片 """
        slice_string = segment.split(":")
        size = len(slice_string)
        if size == 2:
            start = 0 if slice_string[0]=='' else int(slice_string[0])
            end = wave_length if  slice_string[1]=='' else int(slice_string[1])
            stride = 1
        elif size == 3:
            start = 0 if slice_string[0]=='' else int(slice_string[0])
            end = wave_length if  slice_string[1]=='' else int(slice_string[1])
            stride = 1 if  slice_string[2]=='' else int(slice_string[2])
        else:
            raise Exception("slice is error!! (DrawWave class")
        return start, end, stride
    
    
    def plotWave(self, wave, title=None, xlabel="Sample Points", ylabel="Altitude", segment=":"):
        """ 畫單一波型圖  
        Input
        -----------
        segment : 切片，跟list用法一樣，只是放在字串裡面 ex: ":", ":10", "10:20" , "10:" , "10:30:2"
        """
        start, end, stride = self.__segment(segment, len(wave))
        x = [i for i in range(start, end, stride)]
        
        plt.plot(x, wave[start:end:stride])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.show()
    
    def plotWave_withEvents(self, wave, events, title=None, xlabel="Sample Points", ylabel="Altitude", isEvent_1=False, segment=":"):
        """ 畫單一波型圖 + 事件點(on、off合併) 
        Input
        -----------
        segment : 切片，跟list用法一樣，只是放在字串裡面 ex: ":", ":10", "10:20" , "10:" , "10:30:2"
        """
        start, end, stride = self.__segment(segment, len(wave))
        x = [i for i in range(start, end, stride)]
        ## 建立事件波型
        if isEvent_1:
            half_value = 1
        else:
            half_value = max(wave)//2
            half_value = half_value if half_value>1 else 1
        events_y = np.zeros((len(wave)))
        for e in events:
            events_y[e] = half_value
        
        
        plt.plot(x, wave[start:end:stride], c="b", label="wave")
        plt.plot(x, events_y[start:end:stride], c="r", label="event(on && off)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.show()
        
        
    
    def plot_event_detection(self, wave, on_list, off_list, title="", scope=None, save=None):
        if scope and len(scope)==2:
            x = [i for i in range(scope[0], scope[1], 1)]
            plt.plot(x, wave[scope[0]:scope[1]], c='b', label="wave")
            on, off = [], []
            for i in range(len(on_list[0])):
                if(on_list[0][i]>scope[0] and on_list[0][i] < scope[1]):
                    on.append([on_list[0][i], on_list[1][i]])
            for i in range(len(off_list[0])):
                if(off_list[0][i]>scope[0] and off_list[0][i] < scope[1]):
                    off.append([off_list[0][i], off_list[1][i]])
            on, off = np.array(on).T, np.array(off).T        
            plt.plot(on[0], on[1], 'o', c='r', label="On")
            plt.plot(off[0], off[1], '*', c='r', label="Off")
        
        else:
            plt.plot(wave, c='b', label="wave")
            plt.plot(on_list[0], on_list[1], 'o', c='r', label="On")
            plt.plot(off_list[0], off_list[1], '*', c='r', label="Off")
            

        plt.xlabel("period numbers")
        plt.ylabel("I_rms (A)")
        plt.title(title)
        plt.legend(loc="right")
        if save:
            plt.savefig(save)
        plt.show()