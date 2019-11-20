from framework.estimator.base import EstimatorBase
from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge


class Estimator(EstimatorBase):

    def __init__(self, **kwargs):
        EstimatorBase.__init__(self, **kwargs)

    @property
    def default_parameters(self) -> Dict:
        return dict(
            alpha=1.0,
            gamma=0.1
        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return KernelRidge