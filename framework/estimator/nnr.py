from framework.estimator.base import  EstimatorBase
from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn import neighbors

class Estimator(EstimatorBase):

    def __init__( self, **kwargs ):
        EstimatorBase.__init__(self, **kwargs)

    @property
    def default_parameters(self) -> Dict:
        return dict(
            n_neighbors=5,
            weights='distance',  # 'uniform','distance'
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=None
        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return neighbors.KNeighborsRegressor