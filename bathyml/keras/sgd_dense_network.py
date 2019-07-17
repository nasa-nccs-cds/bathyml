from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname(HERE) ), "data" )
from sklearn.preprocessing import normalize
from keras.callbacks import TensorBoard, History
from bathyml.logging import LOG_DIR

class DenseSGDNetwork:

    def __init__(self):
        self.model = None
        self.tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True)

    def build(self, input_dim: int, hiddenLayers: List[int], **kwargs):
        activation = kwargs.get( "activation", "relu" )
        if self.model is None:
            self.model = Sequential()
            for index, hlUnits in enumerate(hiddenLayers):
                if index == 0:  self.model.add( Dense( units=hlUnits, activation=activation, input_dim=input_dim ) )
                else:           self.model.add( Dense( units=hlUnits, activation=activation ) )
            self.model.add( Dense( units=1 ) )
            self.model.compile( loss='mse', optimizer='sgd', metrics=['accuracy'] )


    def train(self, x_training_data: str, y_training_data: str, hiddenLayers: List[int], **kwargs ):
        x_training_data_array = np.loadtxt(x_training_data, delimiter=',')
        x_train = normalize( x_training_data_array[:,0:-1], axis=0 )
        y_training_data_array = np.loadtxt(y_training_data, delimiter=',')
        y_train = normalize( y_training_data_array, axis=0 )
        epochs = kwargs.get( "epochs", 200 )
        batch_size = kwargs.get("batch_size", 1000)
        validation_fraction = kwargs.get( "validation_fraction", 0.25 )
        verbose = kwargs.get( "verbose", 2 )
        net.build( x_train.shape[1], hiddenLayers )
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_fraction, shuffle=True, callbacks=[self.tensorboard] )
        loss_and_metrics = self.model.evaluate( x_train, y_train, batch_size=128)
        print( str(loss_and_metrics) )

if __name__ == '__main__':
    ddir = os.path.join( DATA, "RandomForestTests", "RFA_Outputs", "LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRatios_6b_100_sqrt" )
    x_training_data = os.path.join( ddir, "temp_X_train.csv" )
    y_training_data = os.path.join( ddir, "temp_Y_train.csv")
    net = DenseSGDNetwork()
    net.train( x_training_data, y_training_data, [64] )