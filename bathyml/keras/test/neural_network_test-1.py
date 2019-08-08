import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os, copy, sys, numpy as np, pickle
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
nEpochs = 1000
learningRate=0.01
momentum=0.9
shuffle=False
useValidation = False
decay=0.
nRuns = 1
poly_degree = 1
nesterov=False
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='elu' # 'elu' 'tanh'

def getLayers3( input_dim ):
    return  [   Dense( units=64, activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
                Dense( units=32, activation=activation, kernel_initializer = initWtsMethod ),
                Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers2( input_dim ):
    return [   Dense( units=32, activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
               Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers1( input_dim ):
    return [  Dense( units=1, input_dim=input_dim, kernel_initializer = initWtsMethod )  ]

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
    for layer in getLayers3(nBands): model.add( layer )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
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


if __name__ == '__main__':
    x_train: np.ndarray = read_csv_data( "temp_X_train.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train.csv" )

    if poly_degree > 1:
        poly = PolynomialFeatures(poly_degree)
        poly.fit(x_train)
        x_train = poly.transform(x_train)
    nTrainSamples = x_train.shape[0]

    x_valid: np.ndarray = read_csv_data( "temp_X_test.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test.csv" )
    if poly_degree > 1:
        poly = PolynomialFeatures(poly_degree)
        poly.fit(x_valid)
        x_valid = poly.transform(x_valid)
    nValidSamples = x_valid.shape[0]

    nSamples = x_train.shape[0]
    input_dim = x_train.shape[1]
    validation_fraction = nValidSamples/nSamples if useValidation else 0.0
    xmax, xmin = list(x_train.max(axis=0).tolist()), list(x_train.min(axis=0).tolist())
    ymax, ymin = y_train.max(), y_train.min()
    print( f"InputDim: {input_dim}, #Training samples: {nTrainSamples}, #Validation samples: {nValidSamples}, #Total samples: {nSamples}, validation_fraction: {validation_fraction}, xmax = {xmax}, xmin = {xmin}, ymax = {ymax}, ymin = {ymin}")
    x_train = normalize( x_train )
    y_train = normalize( y_train )
    xmax, xmin = list(x_train.max(axis=0).tolist()), list(x_train.min(axis=0).tolist())
    ymax, ymin = y_train.max(), y_train.min()
    print( f"NORMALIZED xmax = {xmax}, xmin = {xmin}, ymax = {ymax}, ymin = {ymin}")

    ens_min_loss = sys.float_info.max
    best_model_index = None
    min_index = None
    history = {}

    for model_index in range( nRuns ):
        model = get_poly_model(model_index,input_dim) if poly_degree > 1 else get_model(model_index)
        tensorboard = TensorBoard(log_dir=f"{tb_log_dir}/{time()}")
        history[model_index] = model.fit( x_train, y_train, epochs=nEpochs, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard] )
        train_loss = np.array( history[model_index].history['loss'] )
        min_train_loss = train_loss.min(axis=0, initial=sys.float_info.max)
        if validation_fraction > 0:
            val_loss = np.array(history[model_index].history['val_loss'])
            min_val_loss = val_loss.min(axis=0, initial=sys.float_info.max)
            total_loss = val_loss + train_loss
            min_loss = total_loss.min(axis=0, initial=sys.float_info.max)
            print( f"Completed model run {model_index}, min_val_loss={min_val_loss}, min_train_loss={min_train_loss}, min_total_loss={min_loss} ")
        else:
            min_loss = min_train_loss
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
        history = test_model.fit(x_train, y_train, epochs=min_index, validation_split=validation_fraction, verbose=0, shuffle=shuffle, callbacks=[tensorboard])
    else:
        test_model = best_model

    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    ax0 = plt.subplot("211")
    ax0.set_title("Validation Data")
    prediction_valid = test_model.predict( normalize( x_valid ) )
    ax0.plot(range(y_valid.shape[0]), normalize( y_valid ), "b--", label="validation data")
    ax0.plot(range(prediction_valid.shape[0]), prediction_valid, "r--", label="prediction")
    ax0.legend()

    ax1 = plt.subplot("212")
    ax1.set_title("Training Data")
    prediction_train = test_model.predict( normalize( x_train ) )
    ax1.plot(range(y_train.shape[0]), normalize( y_train ), "b--", label="training data")
    ax1.plot(range(prediction_train.shape[0]), prediction_train, "r--", label="prediction")
    ax1.legend()

    saved_model_path = os.path.join( outDir, f"model.{datetime.now().strftime('%m-%d-%H-%M-%S')}")
    filehandler = open(saved_model_path,"wb")
    pickle.dump( test_model.get_weights(), filehandler )
    print( f"Saved weights to file {saved_model_path}" )

    plt.tight_layout()
    plt.show()