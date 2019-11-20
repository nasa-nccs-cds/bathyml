from framework.estimator.base import EstimatorBase
from bathyml.common.data import *
from typing import List, Optional, Tuple, Dict, Any, Type
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

class Estimator(EstimatorBase):

    def __init__( self, **kwargs ):
        EstimatorBase.__init__(self, handles_validation=True, **kwargs )

    @property
    def default_parameters(self) -> Dict:
        return dict(
            hidden_layer_sizes=(32,),
            activation="tanh",
            solver='adam',
            alpha=1e-06,
            batch_size='auto',
            learning_rate="constant",  # ""constant","invscaling"
            learning_rate_init=0.01,
            power_t=0.2,  # 0.5,
            max_iter=100,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.2,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=100 )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return MLPRegressor


    def fit1( self, xdata: np.ndarray, ydata: np.ndarray,  **kwargs ):
        nFolds = kwargs.get( 'nFolds', 5 )
        validFold = kwargs.get('validFold', 4)
        nTrials = kwargs.get('nTrials', 2)
        if self.instance.early_stopping and nFolds > 1:
            if validFold < 0:
                self.update_parameters( validation_fraction=1.0 / nFolds, verbose=True )
                for iTrial in range(nTrials):
                    for iFold in range(nFolds):
                        print( f"\n Executing Fit for trial {iTrial}: fold {iFold}\n")
                        x_train, x_test, y_train, y_test = getKFoldSplit( xdata, ydata, nFolds, iFold )
                        x_data, y_data = np.concatenate( (x_train, x_test) ), np.concatenate( (y_train, y_test) )
                        self.best_validation_score_ = 0
                        self.instance.fit( x_data, y_data, **kwargs )
            else:
                self.update_parameters( validation_fraction=1.0 / nFolds, verbose=False )
                x_train, x_test, y_train, y_test = getKFoldSplit(xdata, ydata, nFolds, validFold )
                x_data, y_data = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
                self.instance.fit( x_data, y_data, **kwargs )
        else:
            EstimatorBase.fit( self, xdata, ydata,  **kwargs )

    def fit( self, xdata: np.ndarray, ydata: np.ndarray,  **kwargs ):
        x_test = kwargs.pop("x_test",None)
        y_test = kwargs.pop("y_test", None)
        if x_test is None:
            x_data, y_data = xdata, ydata
        else:
            x_data, y_data = np.concatenate((xdata, x_test)), np.concatenate((ydata, y_test))
        self.instance.fit( x_data, y_data, **kwargs )


