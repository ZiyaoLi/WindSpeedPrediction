from keras.layers import Input, Conv2D, \
    Reshape, Dropout, Flatten, \
    concatenate, Dense, TimeDistributed, LSTM, SimpleRNN
from keras.models import Model
from keras import optimizers as opts
import numpy as np
import pandas as pd
from make_jiekou_data import h,p,y
from keras.callbacks import EarlyStopping

n_pred = 3
hist_days = 7
hist_points = 9
opt = opts.RMSprop(lr=0.001)

input_hist = Input(shape=(None, hist_days, hist_points))
input_pred = Input(shape=(None, n_pred,))

hist = Reshape(target_shape=(-1, hist_days, hist_points, 1))(input_hist)
hist = TimeDistributed(Conv2D(filters=2, kernel_size=(3, 3), activation='relu'))(hist)
hist = TimeDistributed(Conv2D(filters=2, kernel_size=(3, 3), activation='relu'))(hist)
hist = TimeDistributed(Flatten())(hist)
hist = Dense(units=5, activation='tanh')(hist)
hist = Dropout(0.25)(hist)

pred = Dense(units=10, activation='tanh')(input_pred)
predictor = concatenate([hist, pred])  # B * (5 + 10)

output = LSTM(units=8,
              activation='tanh',
              use_bias=False,
              return_sequences=True,
              stateful=False
              )(predictor)
output = TimeDistributed(Dropout(0.25))(output)
output = TimeDistributed(Dense(units=5, activation='tanh'))(output)
output = TimeDistributed(Dense(units=1, activation='relu'))(output)


model = Model(inputs=[input_hist, input_pred], outputs=output)

model.compile(optimizer=opt, loss='mse')
print("model compiled...")

# example input:
a = h[:,:-96*21,:,:]
b = p[:,:-96*21,:]
c = y[:,:-96*21,:]

a1 = h[:,-96*21:,:,:]
b1 = p[:,-96*21:,:]
c1 = y[:,-96*21:,:]

history = model.fit(x=[a, b], y=c, batch_size=6, epochs=1000, validation_data=[[a1, b1], c1], verbose=2,
                    callbacks=[EarlyStopping(monitor='loss' ,patience=50, verbose=1)])

r = model.predict(x=[a1, b1]).reshape([-1])
l = c1.reshape([-1])
p = np.vstack([l, r]).transpose()
p = pd.DataFrame(p)
p.to_csv("rst\\rst_cnn+rnn.csv", header=None, index=None)

model.save("cnn_rnn_1.h5")

f = open("rst\\hist_cnn+rnn.txt",'w')
f.write(str(history.history))
