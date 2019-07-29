import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os, copy, sys, numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time

models = {}
init_weights = {}

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
nEpochs = 600
learningRate=0.01
momentum=0.9
shuffle=False
decay=0.
nRuns = 5
nesterov=False
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='relu' # 'tanh'
nSegments = 10

print( f"TensorBoard log dir: {tb_log_dir}")

def get_model( index, weights = None ) -> Sequential:
    model = Sequential()
    model.add( Dense( units=32, activation=activation, input_dim=nBands, kernel_initializer = initWtsMethod ) )
    model.add( Dense( units=1, kernel_initializer = initWtsMethod ) )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    if weights is not None:
        model.set_weights( weights )
    else:
        init_weights[index] = [ np.copy( w ) for w in model.get_weights() ]
        models[index] = model
    return model


x_train: np.ndarray = read_csv_data( "temp_X_train.csv", nBands )
y_train: np.ndarray = read_csv_data( "temp_Y_train.csv" )
nTrainSamples = x_train.shape[0]

reindexer = []
for iSeg in range(nSegments): reindexer = reindexer + list( range( iSeg, nTrainSamples, nSegments ) )
x_train_reindexed = x_train[reindexer]
y_train_reindexed = y_train[reindexer]

x_valid: np.ndarray = read_csv_data( "temp_X_test.csv", nBands )
y_valid: np.ndarray = read_csv_data( "temp_Y_test.csv" )
nValidSamples = x_valid.shape[0]

x_train_valid = np.concatenate( (x_train_reindexed,x_valid) )
y_train_valid = np.concatenate( (y_train_reindexed,y_valid) )
nSamples = x_train_valid.shape[0]
validation_fraction = nValidSamples/nSamples
print( f"#Training samples: {nTrainSamples}, #Validation samples: {nValidSamples}, #Total samples: {nSamples}, validation_fraction: {validation_fraction}")

ens_min_loss = sys.float_info.max
best_model_index = None
min_index = None
history = {}

for model_index in range( nRuns ):
    model = get_model(model_index)
    tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
    history[model_index] = model.fit( x_train_valid, y_train_valid, epochs=nEpochs, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard] )
    val_loss = np.array( history[model_index].history['val_loss'] )
    train_loss = np.array( history[model_index].history['loss'] )
    total_loss = val_loss + train_loss
    min_loss = total_loss.min( axis=0, initial=sys.float_info.max )
    if min_loss < ens_min_loss:
        ens_min_loss = min_loss
        min_index = total_loss.tolist().index(min_loss)
        best_model_index = model_index

print( f"Plotting results from model {best_model_index}, N training epocs: {min_index}")
best_model = models[best_model_index]
test_model = get_model( nRuns, init_weights[best_model_index] )
tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
history = test_model.fit(x_train_valid, y_train_valid, epochs=min_index, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard])

fig = plt.figure()
# fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

ax0 = plt.subplot("211")
ax0.set_title("Validation Data")
prediction_valid = best_model.predict( x_valid )
ax0.plot(range(y_valid.shape[0]), y_valid, "b--", label="validation data")
ax0.plot(range(prediction_valid.shape[0]), prediction_valid, "r--", label="prediction")
ax0.legend()

ax1 = plt.subplot("212")
ax1.set_title("Training Data")
prediction_train = best_model.predict( x_train )
ax1.plot(range(y_train.shape[0]), y_train, "b--", label="training data")
ax1.plot(range(prediction_train.shape[0]), prediction_train, "r--", label="prediction")
ax1.legend()

plt.tight_layout()
plt.show()