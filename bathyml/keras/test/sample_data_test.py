from bathyml.keras.training import FitResult, LearningModel
from bathyml.keras.layers import Layer, Layers
import os, numpy as np
outDir = os.path.expanduser("~/results")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
from sklearn.preprocessing import normalize
from bathyml.logging import LOG_DIR
import matplotlib.pyplot as plt
plotPrediction = True
plotVerification = True

nBands = 21
nInterationsPerProc = 1
shuffle = True
batchSize = 32
nEpocs = 100
learnRate = 0.0002
momentum=0.0
decay=0.0
loss_function="mse"
nesterov=False
validation_fraction = 0.20
stopCondition="minVal"
nHiddenUnits = 32
initWtsMethod="lecun_normal"   # lecun_uniform glorot_normal glorot_uniform he_normal lecun_normal he_uniform

ddir = os.path.join(DATA, "csv")
x_training_data = os.path.join(ddir, "temp_X_train.csv")
y_training_data = os.path.join(ddir, "temp_Y_train.csv")
x_training_data_array = np.loadtxt(x_training_data, delimiter=',')[:,:nBands]
y_training_data_array = np.loadtxt(y_training_data, delimiter=',')
x_train = x_training_data_array/x_training_data_array.max(axis=0)
y_train = y_training_data_array/y_training_data_array.max(axis=0)

print( f"Shape of traing data: {x_train.shape}")

layers = [ Layer( "dense", nHiddenUnits, activation = "tanh", kernel_initializer = initWtsMethod ),
           Layer( "dense", nHiddenUnits, activation = "tanh", kernel_initializer = initWtsMethod ),
           Layer( "dense", 1, kernel_initializer = initWtsMethod ) ]

def learning_model_factory( inputs=x_train, target=y_train, weights=None ):
    return LearningModel( inputs, target, layers, batch=batchSize, lrate=learnRate, stop_condition=stopCondition, loss_function=loss_function, momentum=momentum, decay=decay, nesterov=nesterov, epocs=nEpocs, vf=validation_fraction, weights=weights )

result = LearningModel.serial_execute( learning_model_factory, nInterationsPerProc, shuffle )

learningModel = learning_model_factory()
learningModel.plotPrediction( result, "Bathymetry mapping" )
learningModel.plotPerformance( result, "Test" )

