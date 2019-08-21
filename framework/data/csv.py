from typing import List, Optional, Tuple, Dict, Any
import os, copy, sys, pickle, math, numpy as np


class CSVDataHandler:

    def readFeatures( filePath: str, nFeatures: int = 0 ) -> np.ndarray:
        raw_data_array: np.ndarray = np.loadtxt( filePath, delimiter=',')
        if (nFeatures > 0): raw_data_array = raw_data_array[:,:nFeatures]
        return raw_data_array

    def saveFeatures( self, filePath: str, feature_data: np.ndarray ):
        np.savetxt( filePath, feature_data, delimiter="," )