import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels import tsa
import statsmodels.api as sm
from matplotlib import pyplot as plt

train = np.array(pd.read_csv("data.csv", header=None))

N = 100
M = 6

sum_se = np.zeros([300])
n_test = N * M

for mach in range(M):
    train_m = train[:, mach]
    print(mach)
    for i in range(N):
        train0 = train_m[i * 600: i * 600 + 300]
        test0 = train_m[i * 600 + 300: i * 600 + 600]
        r = ARIMA(train0, (1, 1, 1))
        try:
            pred = r.fit(disp=False).predict(start=300, end=599, typ='levels')
        except:
            print("invalid: %d-%d" % (mach, i))
            n_test -= 1
            continue
        if len(pred) != 300:
            print("invalid: %d-%d" % (mach, i))
            n_test -= 1
            continue
        qqqq = abs(test0 - pred) / abs(test0)
        qqqq[qqqq==np.inf]=0

sum_se /= n_test
print(sum_se)




