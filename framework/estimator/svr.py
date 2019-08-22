from framework.estimator.base import  EstimatorBase
from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn import svm


class Estimator(EstimatorBase):

    def __init__(self, **kwargs):
        EstimatorBase.__init__(self, **kwargs)

    @property
    def default_parameters(self) -> Dict:
        return dict(
            C=5.0,
            cache_size=1000,
            coef0=0.0,
            degree=3,
            epsilon=0.1,
            gamma=0.5,
            kernel='rbf',
            max_iter=-1,
            shrinking=True,
            tol=0.001,
            verbose=False
        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return svm.SVR