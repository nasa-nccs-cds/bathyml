from framework.estimator.base import EstimatorBase
from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor

class Estimator(EstimatorBase):

    def __init__(self, **kwargs):
        EstimatorBase.__init__(self, **kwargs)

    @property
    def default_parameters(self) -> Dict:
        return dict(
            kernel=None,
            alpha=5.0,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=5,
            normalize_y=True,
            copy_X_train=True,
            random_state=None
        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return GaussianProcessRegressor