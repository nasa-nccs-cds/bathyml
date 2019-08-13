import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from typing import List, Optional, Tuple, Dict, Any
import os, copy, sys, numpy as np, pickle, math
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.preprocessing import PolynomialFeatures
from time import time
from datetime import datetime

models = {}
init_weights = {}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
outDir = os.path.join(DATA, "results")
if not os.path.exists(outDir): os.makedirs( outDir )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nBands = 21
nEpochs = 250
learningRate=0.01
momentum=0.9
shuffle=True
nLayers = 2
nHidden = [ 64 ]
validation_fraction = 0.2
use_total_error_for_stop = False

decay=0.
nRuns = 10
nesterov=False
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='tanh' # 'tanh'
network_label = f"{nLayers}--{'-'.join([str(ih) for ih in nHidden])}-seg-{validation_fraction:4.2f}"

def getLayers1( input_dim ):
    return [  Dense( units=1, input_dim=input_dim, kernel_initializer = initWtsMethod )  ]

def getLayers2( input_dim ):
    return [   Dense( units=nHidden[0], activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
               Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers3( input_dim ):
    return  [   Dense( units=nHidden[0], activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
                Dense( units=nHidden[1], activation=activation, kernel_initializer = initWtsMethod ),
                Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers4( input_dim ):
    return  [   Dense( units=nHidden[0], activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
                Dense( units=nHidden[1], activation=activation, kernel_initializer = initWtsMethod ),
                Dense( units=nHidden[2], activation=activation, kernel_initializer=initWtsMethod),
                Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers( input_dim ):
    if nLayers == 1: return getLayers1( input_dim )
    if nLayers == 2: return getLayers2( input_dim )
    if nLayers == 3: return getLayers3( input_dim )
    if nLayers == 4: return getLayers4( input_dim )

def read_csv_data( fileName: str, nBands: int = 0 ) -> np.ndarray:
    file_path: str = os.path.join( ddir, fileName )
    raw_data_array: np.ndarray = np.loadtxt( file_path, delimiter=',')
    if (nBands > 0): raw_data_array = raw_data_array[:,:nBands]
    return raw_data_array

def normalize( array: np.ndarray, scale = 1.5 ):
    ave = array.mean( axis=0 )
    std = array.std( axis=0 )
    return (array-ave)/(scale*std)

print( f"TensorBoard log dir: {tb_log_dir}")

def get_model( index, weights = None ) -> Sequential:
    model = Sequential()
    for layer in getLayers(nBands): model.add( layer )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    if weights is not None:
        model.set_weights( weights )
    else:
        init_weights[index] = [ np.copy( w ) for w in model.get_weights() ]
        models[index] = model
    return model

def get_poly_model( index, input_dim, weights = None ) -> Sequential:
    model = Sequential()
    for layer in getLayers1(input_dim): model.add( layer )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    if weights is not None:
        model.set_weights( weights )
    else:
        init_weights[index] = [ np.copy( w ) for w in model.get_weights() ]
        models[index] = model
    return model

def interleave( a0: np.ndarray, a1: np.ndarray ) -> np.ndarray:
    alen = min( a0.shape[0], a1.shape[0] )
    if len( a0.shape ) == 1:
        result = np.empty( ( 2*alen ) )
        result[0::2] = a0[0:alen]
        result[1::2] = a1[0:alen]
    else:
        result = np.empty( ( 2*alen, a0.shape[1] ) )
        result[0::2, :] = a0[0:alen]
        result[1::2, :] = a1[0:alen]
    return result

def smooth( x: np.array, window_len=21 ):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve( w / w.sum(), s, mode='valid' )
    return y

def getTrainingData( x_train, y_train, x_valid, y_valid ) -> Tuple[np.ndarray,np.ndarray]:
    x_train_valid = normalize( interleave(x_train,x_valid) )
    y_train_valid = normalize( interleave(y_train,y_valid) )
    return x_train_valid, y_train_valid

if __name__ == '__main__':
    x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )

    # if poly_degree > 1:
    #     poly = PolynomialFeatures(poly_degree)
    #     poly.fit(x_train)
    #     x_train = poly.transform(x_train)

    x_valid: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )

    # if poly_degree > 1:
    #     poly = PolynomialFeatures(poly_degree)
    #     poly.fit(x_valid)
    #     x_valid = poly.transform(x_valid)


    x_train_valid, y_train_valid = getTrainingData( x_train, y_train, x_valid, y_valid )
    nValidSamples = int( round( x_train_valid.shape[0] * validation_fraction ) )
    nTrainSamples = x_train_valid.shape[0] - nValidSamples
    input_dim = x_train.shape[1]
    vWeight = math.sqrt( 1.0 / nValidSamples )
    tWeight = math.sqrt( 1.0 / nTrainSamples )

    ens_min_loss = sys.float_info.max
    best_model_index = None
    min_index = None
    history = {}

    for model_index in range( nRuns ):
        model = get_model(model_index)
        tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
        history[model_index] = model.fit( x_train_valid, y_train_valid, epochs=nEpochs, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard] )
        train_loss = smooth( np.array( history[model_index].history['loss'] ) )
        if validation_fraction > 0:
            val_loss = smooth( np.array(history[model_index].history['val_loss']) )
            total_loss = vWeight*val_loss + tWeight*train_loss if use_total_error_for_stop else val_loss
            min_loss = total_loss.min(axis=0, initial=sys.float_info.max)
            print( f"Completed model run {model_index}, min_total_loss={min_loss} ")
        else:
            min_loss = train_loss.min(axis=0, initial=sys.float_info.max)
            total_loss = train_loss
            print(f"Completed model run {model_index}, min_loss={min_loss} ")

        if min_loss < ens_min_loss:
            ens_min_loss = min_loss
            min_index = total_loss.tolist().index(min_loss)
            best_model_index = model_index

    print( f"Plotting results from model {best_model_index} (loss={ens_min_loss}), N training epocs: {min_index}")
    best_model = models[best_model_index]
    tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
    if validation_fraction > 0:
        test_model = get_model( nRuns, init_weights[best_model_index] )
        history = test_model.fit(x_train_valid, y_train_valid, epochs=min_index, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard])
    else:
        test_model = best_model

    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    ax0 = plt.subplot("211")
    ax0.set_title( f"{network_label} Training Data")
    prediction_training: np.ndarray = test_model.predict( x_train_valid[:nTrainSamples] )
    xaxis = range(prediction_training.shape[0])
    ax0.plot(xaxis, y_train_valid[:nTrainSamples], "b--", label="validation data")
    ax0.plot(xaxis, prediction_training, "r--", label="prediction")
    ax0.legend()

    ax1 = plt.subplot("212")
    prediction_validation: np.ndarray = test_model.predict( x_train_valid[nTrainSamples:] )
    diff = y_train_valid[nTrainSamples:] - prediction_validation
    mse = math.sqrt( (diff*diff).mean() )
    ax1.set_title( f"{network_label} Validation Data MSE = {mse:.2f} " )
    xaxis = range(prediction_validation.shape[0])
    ax1.plot( xaxis, y_train_valid[nTrainSamples:], "b--", label="training data")
    ax1.plot( xaxis, prediction_validation, "r--", label="prediction")
    ax1.legend()

    saved_results_path = os.path.join(outDir, f"results_{network_label}__{datetime.now().strftime('%m-%d-%H.%M.%S')}")
    filehandler = open(saved_results_path, "wb")
    pickle.dump( ( y_train_valid.tolist(), prediction_training.tolist(), prediction_validation.tolist() ) , filehandler)
    print(f"Saved results to file {saved_results_path}")

    saved_model_path = os.path.join( outDir, f"model_{network_label}__{datetime.now().strftime('%m-%d-%H.%M.%S')}")
    filehandler = open(saved_model_path,"wb")
    pickle.dump( test_model.get_weights(), filehandler )
    print( f"Saved weights to file {saved_model_path}" )

    plt.tight_layout()
    plt.show()