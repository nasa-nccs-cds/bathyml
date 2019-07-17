from keras.models import Sequential
from keras.layers import Dense
import os, numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time

outDir = os.path.expanduser("~/results")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nBands = 21
nEpochs = 50
print( f"TensorBoard log dir: {tb_log_dir}")

x_training_data = os.path.join(ddir, "temp_X_train.csv")
y_training_data = os.path.join(ddir, "temp_Y_train.csv")
x_training_data_array = np.loadtxt(x_training_data, delimiter=',')[:,:nBands]
y_training_data_array = np.loadtxt(y_training_data, delimiter=',')
x_train = x_training_data_array/x_training_data_array.max(axis=0)
y_train = y_training_data_array/y_training_data_array.max(axis=0)

model = Sequential()
tensorboard = TensorBoard( log_dir=f"{tb_log_dir}/{time()}" )

model.add( Dense( units=64, activation='relu', input_dim=nBands ) )
model.add( Dense( units=1 ) )

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

model.fit( x_train, y_train, epochs=nEpochs, callbacks=[tensorboard] )

prediction = model.predict( x_train )

plt.plot(range(y_train.shape[0]), y_train, "b--", label="training data")
plt.plot(range(prediction.shape[0]), prediction, "r--", label="prediction")
plt.show()