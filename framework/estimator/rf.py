from framework.estimator.base import  EstimatorBase
from typing import List, Optional, Tuple, Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor


class Estimator(EstimatorBase):

    def __init__(self, **kwargs):
        EstimatorBase.__init__(self, **kwargs)

    @property
    def default_parameters(self) -> Dict:
        return dict(
            n_estimators=50,
            max_features='sqrt',  # 'log2'
            max_depth=20,
            oob_score=True
        )

    @property
    def _constructor(self) -> Type[BaseEstimator]:
        return RandomForestRegressor