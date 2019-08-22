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
    def new(cls, etype, **parms ) -> "EstimatorBase":
        module = __import__( "framework.estimator." + etype )
        estimator_module = getattr( module.estimator, etype )
        my_class = getattr( estimator_module, "Estimator" )
        return my_class(**parms)

    def fit( self, xdata: np.ndarray, ydata: np.ndarray, validation_fraction: float, *args, **kwargs ):
        x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=validation_fraction, shuffle=False )
        self.instance.fit( x_train, y_train, *args, **kwargs )
        return x_train, x_test, y_train, y_test

    def predict( self, xdata: np.ndarray, *args, **kwargs ) -> np.ndarray:
        return self.instance.predict( xdata, *args, **kwargs )

    def gridSearch( self, xdata: np.ndarray, ydata: np.ndarray, param_grid: List[Dict[str,List]], **kwargs ):
        kwargs["n_jobs"]    = kwargs.get( "n_jobs", -1 )
        kwargs["cv"]        = kwargs.get( "cv", KFold( n_splits=3, shuffle=False ))
        kwargs["scoring"]   = kwargs.get( "scoring", 'neg_mean_squared_error' )
        kwargs["refit"] = kwargs.get( "refit", False )
        gridSearchInstance = GridSearchCV( self.instance, param_grid, **kwargs )
        gridSearchInstance.fit( xdata, ydata )
        print( f"GridSearch: Best PARAMS= {gridSearchInstance.best_params_}" )
        self.update_parameters( **gridSearchInstance.best_params_ )
        return gridSearchInstance.best_params_

    def parameterSearch(self, xdata: np.ndarray, ydata: np.ndarray, param_grid: Dict[str,List], **kwargs ):
        testList = []
        
