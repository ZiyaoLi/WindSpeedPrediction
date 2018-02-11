import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import acf
from statsmodels import tsa
import statsmodels.api as sm
from matplotlib import pyplot as plt

train = np.array(pd.read_csv("data.csv", header=None))

N = 100
M = 6

len21 = 80
len22 = 81
len11 = 153
len12 = 156


sum_se = np.zeros([300])
n_test = N * M

for mach in range(M):
    train_m = train[:, mach]
    print(mach)
    for i in range(N):
        train0 = train_m[i * 600: i * 600 + 300]
        test0 = train_m[i * 600 + 300: i * 600 + 600]
        a2, d2, d1 = pywt.wavedec(train0, 'db4', level=2)
        model_a2 = ARMA(a2, order=(1, 2))
        model_d2 = ARMA(d2, order=(1, 0))
        model_d1 = ARMA(d1, order=(2, 1))
        try:
            rst_a2 = model_a2.fit(disp=False)
            rst_d2 = model_d2.fit(disp=False)
            rst_d1 = model_d1.fit(disp=False)
        except:
            print("invalid: %d-%d" % (mach, i))
            n_test -= 1
            continue
        pa2 = model_a2.predict(params=rst_a2.params, start=1, end=155)
        pd2 = model_d2.predict(params=rst_d2.params, start=1, end=155)
        pd1 = model_d1.predict(params=rst_d1.params, start=1, end=303)
        pa2[:80] = a2[:80]
        pd2[:80] = d2[:80]
        pd1[:80] = d1[:80]
        c = [pa2, pd2, pd1]
        pred = pywt.waverec(c, 'db4')[300: 600]
        sum_se += abs(test0 - pred) / abs(test0)
sum_se /= n_test
print(sum_se)




