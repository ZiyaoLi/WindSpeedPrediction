from keras.layers import Input, Conv2D, \
    Reshape, Dropout, Flatten, \
    concatenate, Dense
from keras.models import Model
from keras import optimizers as opts
import numpy as np
import pandas as pd
from data import h,p,y
from keras.callbacks import EarlyStopping

# simple OLS: mse: 3.17; mae: 1.34

n_pred = 9
hist_days = 7
hist_points = 9
opt = opts.RMSprop(lr=0.0005)

input_hist = Input(shape=(hist_days, hist_points))
input_pred = Input(shape=(n_pred,))

hist = Reshape(target_shape=(hist_days, hist_points, 1))(input_hist)
hist = Conv2D(filters=4, kernel_size=(3, 3), activation='tanh')(hist)
hist = Conv2D(filters=4, kernel_size=(3, 3), activation='tanh')(hist)
hist = Flatten()(hist)
hist = Dense(units=8, activation='relu')(hist)
hist = Dropout(0.4)(hist)

pred = Dense(units=10, activation='tanh')(input_pred)
predictor = concatenate([hist, pred])  # B * (30 + 3)

output = Dense(units=10, activation='tanh')(predictor)
output = Dropout(0.3)(output)
output = Dense(units=1, activation='relu')(output)

model = Model(inputs=[input_hist, input_pred], outputs=output)

model.compile(optimizer=opt, loss='mae')

# example input
a = h[:,:-96*21,:,:].reshape([-1, hist_days, hist_points])
b = p[:,:-96*21,:].reshape([-1, n_pred])
c = y[:,:-96*21,:].reshape([-1, 1])

a1 = h[:,-96*21:,:,:].reshape([-1, hist_days, hist_points])
b1 = p[:,-96*21:,:].reshape([-1, n_pred])
c1 = y[:,-96*21:,:].reshape([-1, 1])

history = model.fit(x=[a, b], y=c, batch_size=1000, epochs=200000,validation_data=[[a1, b1], c1], verbose=2,
                    callbacks=[EarlyStopping(monitor='loss', patience=500, verbose=1)])

test_rst = model.evaluate(x=[a1, b1], y=c1)
print(str(test_rst))

r = model.predict(x=[a1, b1]).reshape([-1])
l = c1.reshape([-1])
p = np.vstack([l, r]).transpose()
p = pd.DataFrame(p)
p.to_csv("rst\\rst_cnn+mlp_adjustedhist8.csv", header=None, index=None)


model.save("cnn_mlp_adjustedhist8.h5")

f = open("rst\\hist_cnn+mlp_adjustedhist8.txt",'w')
f.write(str(history.history))
