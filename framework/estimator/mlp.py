from framework.estimator.base import EstimatorBase
from bathyml.common.data import *
from typing import List, Optional, Tuple, Dict, Any, Type
import csv, numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

class Estimator(EstimatorBase):

    def __init__( self, **kwargs ):
        EstimatorBase.__init__(self, handles_validation=True, **kwargs )
        self.init_weights = None
        self.final_weights = None
        self.init_biases = None
        self.final_biases = None

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


    def fit( self, xdata: np.ndarray, ydata: np.ndarray,  **kwargs ):
        x_test = kwargs.pop("x_test",None)
        y_test = kwargs.pop("y_test", None)
        regressor: MLPRegressor = self.instance
        if x_test is None:
            x_data, y_data = xdata, ydata
        else:
            x_data, y_data = np.concatenate((xdata, x_test)), np.concatenate((ydata, y_test))
        self.init( x_data, y_data,  **kwargs )
        regressor.fit( x_data, y_data )
        self.final_weights = [ np.copy( w ) for w in regressor.coefs_  ]
        self.final_biases =  [ np.copy( w ) for w in regressor.intercepts_ ]
        regressor.warm_start = False
        write_final_weights = kwargs.get('write_final_weights', None)
        if write_final_weights is not None:
            self.write_weights( write_final_weights, self.final_weights + self.final_biases )

    def write_weights( self, outfile_path, weights_data ):
        with open(outfile_path, "wb") as outfile:
            print(f"Write weights_data to file {outfile_path}")
            pickle.dump( weights_data, outfile )

    def init(self, X, y, **kwargs ):
        from sklearn.utils import check_random_state
        incremental = kwargs.get('incremental', False )
        regressor: MLPRegressor = self.instance
        hidden_layer_sizes = regressor.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        regressor._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %  hidden_layer_sizes)

        X, y = regressor._validate_input(X, y, incremental)
        n_samples, n_features = X.shape

        if y.ndim == 1: y = y.reshape((-1, 1))
        regressor.n_outputs_ = y.shape[1]
        layer_units = ( [n_features] + hidden_layer_sizes + [regressor.n_outputs_] )
        regressor._random_state = check_random_state( regressor.random_state )
        regressor._initialize(y, layer_units)
        self.init_weights = [ np.copy( w ) for w in regressor.coefs_  ]
        self.init_biases =  [ np.copy( w ) for w in regressor.intercepts_ ]
        regressor.warm_start = True
        write_init_weights = kwargs.get('write_init_weights', None)
        if write_init_weights is not None:
            self.write_weights( write_init_weights, self.init_weights + self.init_biases )



