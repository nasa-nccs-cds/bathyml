from bathyml.common.data import *
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from time import time
from datetime import datetime
from sklearn import preprocessing
from framework.estimator.base import EstimatorBase
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

modelType = "svr"
validation_fraction = 0.2

if __name__ == '__main__':
    print("Reading Data")
    input_x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    input_y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    input_x_test: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    input_y_test: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )
    x_data, y_data = getTrainingData( input_x_train, input_y_train, input_x_test, input_y_test )
    x_data_norm = preprocessing.scale( x_data )

    estimator: EstimatorBase = EstimatorBase.new( modelType )
    print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )

    param_grids = dict(
        mlp = [ dict( learning_rate_init = [ 0.005, 0.01, 0.02 ], alpha = [ 5e-07, 1e-06, 5e-06 ] ) ],
        rf = [ dict( n_estimators=[20,30,40,50],  max_depth=[5,10,15,20]  ) ],
        svr=[ dict(  C = [5.0], gamma=[0.3,0.4,0.5] ) ],
        nnr = [dict(n_neighbors=[100,150,200], weights=['uniform','distance'])]
    )

    best_params = estimator.gridSearch( x_data_norm, y_data, param_grids[modelType] )
    x_train, x_test, y_train, y_test = estimator.fit( x_data_norm, y_data, validation_fraction )

    test_prediction = estimator.predict(x_test)
    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    mse =  math.sqrt( mean_squared_error( y_test, test_prediction ) )
    ax0 = plt.subplot()
    ax0.set_title( f"{modelType} Test Data MSE = {mse:.2f} ")
    xaxis = range(test_prediction.shape[0])
    ax0.plot(xaxis, y_test, "b--", label="test data")
    ax0.plot(xaxis, test_prediction, "r--", label="prediction")
    ax0.legend()

    plt.tight_layout()
    plt.show()