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

modelType = "mlp"
nFolds = 5
validFold = nFolds-1

if __name__ == '__main__':
    print("Reading Data")
    input_x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    input_y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    input_x_test: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    input_y_test: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )
    x_data_raw, y_data = getTrainingData( input_x_train, input_y_train, input_x_test, input_y_test )
    x_data_norm = preprocessing.scale( x_data_raw )

    estimator: EstimatorBase = EstimatorBase.new( modelType )
    print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )

    param_grids = dict(
        mlp = [ dict( max_iter=[80, 100, 120, 140, 160], learning_rate=["invscaling","constant"] ) ],
        rf = [ dict( n_estimators=[30],  max_depth=[10]  ) ],
        svr=[ dict(  C = [5.0], gamma=[0.05] ) ],
        nnr = [dict(n_neighbors=[200], weights=['uniform','distance'])]
    )

    best_params = estimator.gridSearch( x_data_norm, y_data, param_grids[modelType] )
    x_train, x_test, y_train, y_test = getKFoldSplit( x_data_norm, y_data, nFolds,  validFold )
    estimator.fit( x_train, y_train )
#    estimator.fit( x_data_norm, y_data )

    test_prediction = estimator.predict(x_test)
    fig = plt.figure()

    mse =  math.sqrt( mean_squared_error( y_test, test_prediction ) )
    ax0 = plt.subplot()
    ax0.set_title( f"{modelType} Test Data MSE = {mse:.2f} ")
    xaxis = range(test_prediction.shape[0])
    ax0.plot(xaxis, y_test, "b--", label="test data")
    ax0.plot(xaxis, test_prediction, "r--", label="prediction")
    ax0.legend()

    plt.tight_layout()
    plt.show()