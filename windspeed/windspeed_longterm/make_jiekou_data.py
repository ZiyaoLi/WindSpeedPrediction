#coding:utf-8
import os
import numpy as np
import pandas as pd

#path
hist = "hist_data.csv"
pred = "total_predicting_data_spline2_"
# pred = "pred_data_"

#initialization
N_days = 7
N_nn = 9
N_pred = 3
N_near = int((N_nn-1)/2)

#initiate data
hist_data = pd.read_csv(hist,header = None)
len_data = len(hist_data)
h = np.zeros([6,len_data-96*N_days,N_days,N_nn])
p = np.zeros([6,len_data-96*N_days,N_pred])

#make label data
y = hist_data[96*N_days:len_data]
y = y.T
y = np.array(y).reshape([6,len_data-96*N_days,1])

#make history data
hist_data = np.array(hist_data)
for i in range(96*N_days,len_data):
    for j in range(1,N_days+1):
        if i - j * 96 >= N_near:
            temp = hist_data[i-j*96-N_near:i-j*96+N_near+1,:]
            temp = temp.T
            h[:,i-96*N_days,j-1,:] = temp
        else:
            tt = i-j*96
            temp = hist_data[i - j * 96 - tt:i - j * 96 + N_near + 1, :]
            temp = temp.T
            h[:,i-96*N_days,j-1,N_near-tt:] = temp
            for k in range(6):
                h[k, i - 96 * N_days, j-1, :N_near - tt] = hist_data[0, k]


#make predict data
for i in range(6):
    pred_data = pd.read_csv(pred+str(i+1)+".csv",header = None)
    pred_data = np.array(pred_data)
    p[i,:,:] = pred_data[96*N_days:len_data,:]



