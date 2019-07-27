from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
nEpochs = 500
learningRate=0.01
momentum=0.9
shuffle=False
validation_fraction=0.2
decay=0.
nRuns = 10
nesterov=True
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='relu' # 'tanh'
print( f"TensorBoard log dir: {tb_log_dir}")

x_training_data: str = os.path.join(ddir, "temp_X_train.csv")
y_training_data: str = os.path.join(ddir, "temp_Y_train.csv")
x_training_data_array: np.ndarray = np.loadtxt(x_training_data, delimiter=',')[:,:nBands]
y_training_data_array: np.ndarray = np.loadtxt(y_training_data, delimiter=',')
x_train: np.ndarray = x_training_data_array/x_training_data_array.max(axis=0)
y_train: np.ndarray = y_training_data_array/y_training_data_array.max(axis=0)

tensorboard = TensorBoard( log_dir=f"{tb_log_dir}/{time()}" )
sgd = SGD( learningRate, momentum, decay, nesterov )
index_permutation = np.random.permutation( x_train.shape[0] )
ens_min_val_loss = 10000
best_model_index = None
models = {}

for model_index in range( nRuns ):

    model = Sequential()
    model.add( Dense( units=32, activation=activation, input_dim=nBands, kernel_initializer = initWtsMethod ) )
    model.add( Dense( units=16, activation=activation, kernel_initializer = initWtsMethod ) )
    model.add( Dense( units=1, kernel_initializer = initWtsMethod ) )
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    x_train_resorted = x_train[ index_permutation ]
    y_train_resorted = y_train[ index_permutation ]

    history = model.fit( x_train_resorted, y_train_resorted, epochs=nEpochs, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard] )

    val_loss = np.array( history.history['val_loss'] )
    final_val_loss = val_loss[-1]
    min_val_loss = val_loss.min( axis=0, initial=10000 )
    models[ model_index ] = ( model, final_val_loss, min_val_loss )
    if min_val_loss < ens_min_val_loss:
        ens_min_val_loss = min_val_loss
        best_model_index = model_index

( best_model, final_val_loss, min_val_loss ) = models[best_model_index]
print( f"best_model_index = {best_model_index}, final_val_loss={final_val_loss}, min_val_loss={min_val_loss}")
prediction = best_model.predict( x_train )
plt.plot(range(y_train.shape[0]), y_train, "b--", label="training data")
plt.plot(range(prediction.shape[0]), prediction, "r--", label="prediction")
plt.show()