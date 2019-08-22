from framework.estimator.base import EstimatorBase
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
            learning_rate="invscaling",  # ""constant",
            learning_rate_init=0.01,
            power_t=0.1,  # 0.5,
            max_iter=500,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.2,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return MLPRegressor

    def fit( self, xdata: np.ndarray, ydata: np.ndarray, validation_fraction: float, *args, **kwargs ):
        self.update_parameters( validation_fraction=validation_fraction )
        self.instance.fit( xdata, ydata, *args, **kwargs )
        x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=validation_fraction, shuffle=False)
        return x_train, x_test, y_train, y_test
