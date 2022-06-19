# -*- coding: utf-8 -*-

import yuyi_nilm_package as yuyi


datas, infos = yuyi.datasets.ReadPLAID_2018().getAggregated()

size = len(datas)
#%%
for i in range(size):
    original_v_wave, original_i_wave = datas[i][0], datas[i][1]
    waveObj = yuyi.Wave().init_wave(original_v_wave, original_i_wave)
    
    break


