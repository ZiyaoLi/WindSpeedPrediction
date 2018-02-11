import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt

train = np.array(pd.read_csv("data.csv", header=None))
train = train[0:600, 0]

r = ARIMA(train[:300], (1, 1, 1))
pred = np.zeros([600])
pred[0] = train[0]
pred[1:] = r.fit(disp=False).predict(start=1, end=599, typ='levels')

a2, d2, d1 = pywt.wavedec(train[:300], 'db4', level=2)
model_a2 = ARMA(a2, order=(1, 2))
model_d2 = ARMA(d2, order=(1, 0))
model_d1 = ARMA(d1, order=(2, 1))
rst_a2 = model_a2.fit(disp=False)
rst_d2 = model_d2.fit(disp=False)
rst_d1 = model_d1.fit(disp=False)
pa2 = model_a2.predict(params=rst_a2.params, start=1, end=155)
pd2 = model_d2.predict(params=rst_d2.params, start=1, end=155)
pd1 = model_d1.predict(params=rst_d1.params, start=1, end=303)
c = [pa2, pd2, pd1]
pred2 = pywt.waverec(c, 'db4')

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(train[:340], "blue")
plt.plot(pred[:340], 'red')
plt.axvline(x=300, ymin=0, ymax=20)
plt.title("model_ARIMA")

plt.subplot(2, 1, 2)
plt.plot(train[:340], "blue")
plt.plot(pred2[:340], 'red')
plt.axvline(x=300, ymin=0, ymax=20)
plt.title("model_WT-ARMA")
