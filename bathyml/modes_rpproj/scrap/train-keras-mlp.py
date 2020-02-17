from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os, copy, sys, numpy as np, pickle
from bathyml.common.data import read_csv_data
from time import time

models = {}
init_weights = {}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nEpochs = 1000
learningRate=0.01
momentum=0.9
shuffle=True
decay=0.
nRuns = 8
nesterov=False
initWtsMethod="glorot_uniform"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform
activation='tanh' # 'tanh'

def getLayers3( input_dim ):
    return  [   Dense( units=32, activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
                Dense( units=8, activation=activation, kernel_initializer = initWtsMethod ),
                Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers4( input_dim ):
    return  [   Dense( units=64, activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
                Dense( units=32, activation=activation, kernel_initializer = initWtsMethod ),
                Dense( units=8, activation=activation, kernel_initializer=initWtsMethod),
                Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers2( input_dim ):
    return [   Dense( units=64, activation=activation, input_dim=input_dim, kernel_initializer = initWtsMethod ),
               Dense( units=1, kernel_initializer = initWtsMethod )  ]

def getLayers1( input_dim ):
    return [  Dense( units=1, input_dim=input_dim, kernel_initializer = initWtsMethod )  ]

def get_model( index, input_size, weights = None ) -> Sequential:
    model = Sequential()
    for layer in getLayers2(input_size): model.add( layer )
    sgd = SGD(learningRate, momentum, decay, nesterov)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    if weights is not None:
        model.set_weights( weights )
    else:
        init_weights[index] = [ np.copy( w ) for w in model.get_weights() ]
        models[index] = model
    return model

def getActivationBackProjection( model_index: int, model: Sequential, target_value, iHiddenUnit = 0, learning_rate = 0.002 ) -> np.ndarray:
    import keras.backend as K
    out_diff = K.mean( (model.layers[-1].output - target_value ) ** 2 ) if iHiddenUnit is None else  K.abs( model.layers[-2].output[0, iHiddenUnit] - target_value )
    grad = K.gradients(out_diff, [model.input])[0]
    grad /= K.maximum(K.sqrt(K.mean(grad ** 2)), K.epsilon())
    iterate = K.function( [model.input, K.learning_phase()], [out_diff, grad] )
    input_img_data = np.zeros( shape=model.weights[0].shape[::-1] )
    print(f"Back Projection Map, model_index = {model_index}, Iterations:")
    out_loss = 0.0
    for i1 in range(5):
        for i2 in range(500):
            out_loss, out_grad = iterate([input_img_data, 0])
            input_img_data -= out_grad * learning_rate
            if out_loss < 0.01:
                print("  --> Converged, niters = " + str(i1*500+i2) + ": loss = " + str(out_loss))
                break
        if( out_loss > 1.0 ):
            learning_rate = learning_rate * 2.0
            print("    ** Doubling Learning Rate: niters = " + str( (i1+1) * 500 ) + ": loss = " + str(out_loss))
        elif out_loss < 0.01: break

    return input_img_data[0]

if __name__ == '__main__':
    pts_data, x_data_raw, y_train_valid = read_csv_data( "pts_merged_final.csv" )
    x_train_valid = normalize(x_data_raw )
    input_dim = x_train_valid.shape[1]
    history = {}
    lines = []

    for model_index in range( nRuns ):
        model = get_model( model_index, input_dim )
        print( f"Max depth: {y_train_valid.max()}")
        history[model_index] = model.fit( x_train_valid, y_train_valid, epochs=nEpochs, verbose=0, shuffle=shuffle )
        train_loss = np.array( history[model_index].history['loss'] )
        min_train_loss = train_loss.min(axis=0, initial=sys.float_info.max)
        min_loss = min_train_loss
        total_loss = train_loss
        print(f"Completed model run {model_index}, min_loss={min_loss} ")

        input_bpvals_bot: np.ndarray = getActivationBackProjection( model_index, model, 21.0 )
        lines.append( ','.join(f'{x:.3f}' for x in input_bpvals_bot) + "\n")

        saved_model_path = os.path.join(outDir, f"mlp_weights_{model_index}")
        filehandler = open(saved_model_path, "wb")
        weights = model.get_weights()
        pickle.dump(weights, filehandler)
        print(f"Saved weights to file {saved_model_path}")

    saved_bpvals_path = os.path.join(outDir, f"bpvals.mlp.csv")
    print( f"Saving bpvals to {saved_bpvals_path}")
    filehandler = open(saved_bpvals_path, "w")
    filehandler.writelines( lines )
    filehandler.close()


