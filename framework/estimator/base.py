from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import numpy as np
import importlib, abc

class EstimatorBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.instance: BaseEstimator = None
        self.handles_validation = kwargs.get( 'handles_validation', False )
        self.parms = kwargs
        self.update_parameters( **kwargs )

    def update_parameters(self, **kwargs ):
        instance_parameters: Dict = self._generateUpdatedParameters( **kwargs )
        self.instance = self._constructor( **instance_parameters )

    def _generateUpdatedParameters(self, **kwargs ) -> Dict:
        parms = dict( **self.default_parameters )
        for key,value in kwargs.items():
            if key in self.default_parameters:
                parms[key] = value
        return parms

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

    def fit( self, xfit: np.ndarray, yfit: np.ndarray, *args, **kwargs ):
        self.instance.fit( xfit, yfit, *args, **kwargs )

    def predict( self, xdata: np.ndarray, *args, **kwargs ) -> np.ndarray:
        return self.instance.predict( xdata, *args, **kwargs )

    def gridSearch( self, xdata: np.ndarray, ydata: np.ndarray, param_grid: List[Dict[str,List]], **kwargs ):
        kwargs["n_jobs"] = kwargs.get("n_jobs",-1)
        kwargs["cv"] = kwargs.get("cv", 5)
        gridSearch = GridSearchCV( self.instance, param_grid, **kwargs )
        gridSearch.fit( xdata, ydata )
        return gridSearch.best_params_
