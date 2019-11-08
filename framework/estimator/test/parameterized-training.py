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

modelType = "rf"
nFolds = 5
validFold = 1 # nFolds-1
trainWithFullDataset = False

if __name__ == '__main__':
    print("Reading Data")
    x_data_raw, y_data = read_csv_data( "pts_merged_final.csv"  )
    x_data_norm = preprocessing.scale( x_data_raw )

    estimator: EstimatorBase = EstimatorBase.new( modelType )
    print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )

    parameters = dict(
        mlp = dict( max_iter=200, learning_rate="constant", early_stopping=True, solver="lbfgs", validation_fraction=1.0/nFolds ),
        rf =  dict( n_estimators=30,  max_depth=10  ) ,
        svr=  dict( C =5.0, gamma=0.05 ) ,
        nnr = dict( n_neighbors=200, weights='distance' )
    )

    x_train, x_test, y_train, y_test = getKFoldSplit( x_data_norm, y_data, nFolds, validFold )
    estimator.update_parameters( **parameters[modelType] )
    if trainWithFullDataset:    estimator.fit( x_data_norm, y_data  )
    else:                       estimator.fit( x_train,     y_train )

    test_prediction = estimator.predict(x_test)
    train_prediction = estimator.predict(x_train)
    fig = plt.figure()

    ax0 = plt.subplot("211")
    mse =  math.sqrt( mean_squared_error( y_train, train_prediction ) )
    ax0.set_title( f"{modelType} Train Data MSE = {mse:.2f} ")
    xaxis = range(train_prediction.shape[0])
    ax0.plot(xaxis, y_train, "b--", label="train data")
    ax0.plot(xaxis, train_prediction, "r--", label="prediction")
    ax0.legend()

    ax1 = plt.subplot("212")
    mse =  math.sqrt( mean_squared_error( y_test, test_prediction ) )
    ax1.set_title( f"{modelType} Test Data MSE = {mse:.2f} ")
    xaxis = range(test_prediction.shape[0])
    ax1.plot(xaxis, y_test, "b--", label="test data")
    ax1.plot(xaxis, test_prediction, "r--", label="prediction")
    ax1.legend()

    plt.tight_layout()
    plt.show()