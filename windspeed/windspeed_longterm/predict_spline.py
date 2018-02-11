#coding:utf-8
import numpy as np
import pandas as pd
from scipy import interpolate as inp
from sklearn.metrics import mean_absolute_error as mae_
from sklearn.metrics import mean_squared_error as mse_


#read data
file = "rst\\rst_cnn+mlp_300.csv"
data = pd.read_csv(file,header = None)
data = np.array(data)
data1 = data
for i in range(6):
    d = data[i*96*21:(i+1)*96*21,:]
    label = d[:,0]
    pred = d[:,1]
    x = range(96*21)
    m = inp.UnivariateSpline(x, pred, s=len(pred)/4)
    y_spline = m(x)
    data1[i*96*21:(i+1)*96*21,1] = y_spline
    print(i+1)
    print(mse_(label, pred))
    print(mse_(label, y_spline))
print(mae_(data1[:,0],data1[:,1]))
data1 = pd.DataFrame(data1)
data1.to_csv("rst\\rst_cnn+mlp_adjustedhist_spline.csv",header = None,index = None)
