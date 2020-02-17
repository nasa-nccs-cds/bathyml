from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import numpy as np
import importlib, abc

class EstimatorBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.instance: BaseEstimator = None
        self.instance_parameters = None
        self.gridSearchInstance: GridSearchCV = None
        self.parms = kwargs
        self.initialize_parameters()
        self.update_parameters( **kwargs )

    def update_parameters(self, **kwargs ):
        for key,value in kwargs.items():
            if key in self.default_parameters:
                self.instance_parameters[key] = value
        if self.instance is None:   self.instance = self._constructor( **self.instance_parameters )
        else:                       self.instance.set_params( **self.instance_parameters )

    def initialize_parameters(self):
        self.instance_parameters = dict( **self.default_parameters )
        self.instance = self._constructor(**self.instance_parameters)

    @property
    def parameterList(self) -> List[str]:
        return self.instance.get_params().keys()

    @property
    @abc.abstractmethod
    def default_parameters(self) -> Dict: pass

    @property
    @abc.abstractmethod
    def _constructor(self) -> Type[BaseEstimator]: pass

    @classmethod
    def mean_squared_error(cls, x: np.ndarray, y: np.ndarray):
        diff = x - y
        return np.sqrt(np.mean(diff * diff, axis=0))

    @classmethod
    def normalize( cls, x: np.ndarray, axis=0, scale = 1.5 ):
        x0 = x - x.mean( axis=axis, keepdims=True )
        mag = x0.std( axis=axis, keepdims=True )
        return x0 / (mag*scale)

    @classmethod
    def new(cls, etype, **parms ) -> "EstimatorBase":
        module = __import__( "framework.estimator." + etype )
        estimator_module = getattr( module.estimator, etype )
        my_class = getattr( estimator_module, "Estimator" )
        return my_class(**parms)

    def shuffle_data(self, input_data, training_data):  # -> ( shuffled_input_data, shuffled_training_data):
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        shuffled_input_data, shuffled_training_data = np.copy(input_data), np.copy(training_data)
        shuffled_input_data[:] = input_data[indices]
        shuffled_training_data[:] = training_data[indices]
        return (shuffled_input_data, shuffled_training_data)

    def fit( self, xdata: np.ndarray, ydata: np.ndarray,  **kwargs ):
        self.instance.fit( *self.shuffle_data( xdata, ydata ), **kwargs )

    def feature_importance( self, xdata: np.ndarray, ydata: np.ndarray,  **kwargs ):
        prediction = self.instance.predict(xdata)
        mse0 =  self.mean_squared_error( ydata, prediction )
        importance = []
        for iFeature in range( xdata.shape[1] ):
            xdata1 = self.shuffle_feature( xdata, iFeature )
            prediction1 = self.instance.predict(xdata1)
            mse1 = self.mean_squared_error(ydata, prediction1)
            importance.append( mse1-mse0 )
        return np.array( importance )

    def predict( self, xdata: np.ndarray, *args, **kwargs ) -> np.ndarray:
        return self.instance.predict( xdata, *args, **kwargs )

    def gridSearch( self, xdata: np.ndarray, ydata: np.ndarray, param_grid: List[Dict[str,List]], nFolds=5, **kwargs ):
        kwargs["n_jobs"]    = kwargs.get( "n_jobs", -1 )
        kwargs["cv"]        = kwargs.get( "cv", KFold( n_splits=nFolds, shuffle=False ))
        kwargs["scoring"]   = kwargs.get( "scoring", 'neg_mean_squared_error' )
        kwargs["refit"] = kwargs.get( "refit", False )
        gridSearchInstance = GridSearchCV( self.instance, param_grid, **kwargs )
        gridSearchInstance.fit( xdata, ydata )
        print( f"GridSearch: Best PARAMS= {gridSearchInstance.best_params_}" )
        self.update_parameters( **gridSearchInstance.best_params_ )
        return gridSearchInstance.best_params_

    def parameterSearch(self, xdata: np.ndarray, ydata: np.ndarray, param_grid: Dict[str,List], **kwargs ):
        testList = []
        
