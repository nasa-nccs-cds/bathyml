from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os, sys, numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time

def read_csv_data( fileName: str, nBands: int = 0 ) -> np.ndarray:
    file_path: str = os.path.join( ddir, fileName )
    raw_data_array: np.ndarray = np.loadtxt( file_path, delimiter=',')
    if (nBands > 0): raw_data_array = raw_data_array[:,:nBands]
    return raw_data_array / raw_data_array.max(axis=0)

outDir = os.path.expanduser("~/results")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nBands = 21
nEpochs = 2000
learningRate=0.01
momentum=0.9
shuffle=False
decay=0.
nRuns = 1
nesterov=False
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='relu' # 'tanh'
print( f"TensorBoard log dir: {tb_log_dir}")

x_train: np.ndarray = read_csv_data( "temp_X_train.csv", nBands )
y_train: np.ndarray = read_csv_data( "temp_Y_train.csv" )
nTrainSamples = x_train.shape[0]

x_valid: np.ndarray = read_csv_data( "temp_X_valid.csv", nBands )
y_valid: np.ndarray = read_csv_data( "temp_Y_valid.csv" )
nValidSamples = x_valid.shape[0]

x_train_valid = np.concatenate( (x_train,x_valid) )
y_train_valid = np.concatenate( (y_train,y_valid) )
nSamples = x_train_valid.shape[0]
validation_fraction = nValidSamples/nTrainSamples

ens_min_val_loss = sys.float_info.max
best_model_index = None
models = {}

for model_index in range( nRuns ):

    model = Sequential()
    model.add( Dense( units=24, activation=activation, input_dim=nBands, kernel_initializer = initWtsMethod ) )
    model.add( Dense( units=1, kernel_initializer = initWtsMethod ) )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
    history = model.fit( x_train_valid, y_train_valid, epochs=nEpochs, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard] )

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
plt.plot(range(y_train.shape[0]), y_train, "b--", label="validation data")
plt.plot(range(prediction.shape[0]), prediction, "r--", label="prediction")
plt.show()