from make_jiekou_data import p,y
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

p1 = p[:, :-96*21, 0]
y1 = y[:, :-96*21, :]
p1 = p1.reshape([-1, 1])
y1 = y1.reshape([-1])
a = lm.LinearRegression()
a.fit(p1, y1)
p2 = p[:, -96*21:, 0]
y2 = y[:, -96*21:, :]
p2 = p2.reshape([-1, 1])
y2 = y2.reshape([-1])
pp = a.predict(p2)
print(mae(y2, pp))
p22 = p2.reshape([-1])
print(mse(p22, y2))

2.54, 2.18, 3.32, 4.12, 2.55, 4.55