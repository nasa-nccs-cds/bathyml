from typing import List, Optional, Tuple, Dict, Any
import os, copy, sys, numpy as np, pickle, math
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname(HERE) ), "data" )
outDir = os.path.join(DATA, "results")
if not os.path.exists(outDir): os.makedirs( outDir )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nBands = 21

class NormalizedArray:

    def __init__(self, array: np.ndarray ):
        self._ave: np.ndarray =  array.mean( axis=0 )
        self._std: np.ndarray = array.std( axis=0 )
        self._raw: np.ndarray = array
        self._data: np.ndarray =(array-self._ave)/self._std

    def data(self) -> np.ndarray:
        return self._data

    def raw(self) -> np.ndarray:
        return self._raw

    def seg(self, seg_fraction: float, first: bool, rescaled: bool = True ) -> np.ndarray:
        NSamples = int( round( seg_fraction * self._data.shape[0] ) )
        if rescaled: return self._data[:NSamples] if first else self._data[NSamples:]
        else:        return self._raw[:NSamples]  if first else self._raw[NSamples:]

    def rescale(self, array: np.ndarray ) -> np.ndarray:
        return array*self._std + self._ave

def read_csv_data( fileName: str ) -> Tuple[np.ndarray,np.ndarray]:
    import csv
    file_path: str = os.path.join( ddir, fileName )
    with open(file_path) as csvfile:
        csvData = csv.reader( csvfile, delimiter=',' )
        headers = None
        ydata = []
        xdata = []
        for index,row in enumerate(csvData):
            if index == 0: headers = row
            else:
                ydata.append( float(row[1]) )
                xdata.append( [ float(r) for r in row[3:]] )
        np_xdata = np.array( xdata )
        np_ydata = np.array( ydata )
        return np_xdata, np_ydata

def normalize( array: np.ndarray ):
    ave = array.mean( axis=0 )
    std = array.std( axis=0 )
    return (array-ave)/std, ave, std

def rescale( array: np.ndarray, ave, std ):
    return array*std + ave

def interleave( a0: np.ndarray, a1: np.ndarray ) -> np.ndarray:
    alen = min( a0.shape[0], a1.shape[0] )
    if len( a0.shape ) == 1:
        result = np.empty( ( 2*alen ) )
        result[0::2] = a0[0:alen]
        result[1::2] = a1[0:alen]
    else:
        result = np.empty( ( 2*alen, a0.shape[1] ) )
        result[0::2, :] = a0[0:alen]
        result[1::2, :] = a1[0:alen]
    return result

def smooth( x: np.array, window_len=21 ):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve( w / w.sum(), s, mode='valid' )
    return y

def getTrainingNArray( x_train, y_train, x_valid, y_valid ) -> Tuple[NormalizedArray,NormalizedArray]:
    x_train_valid = NormalizedArray( interleave(x_train,x_valid) )
    y_train_valid = NormalizedArray( interleave(y_train,y_valid) )
    return x_train_valid, y_train_valid

def getTrainingData( x_train, y_train, x_valid, y_valid ) -> Tuple[np.ndarray,np.ndarray]:
    x_train_valid = interleave(x_train,x_valid)
    y_train_valid = interleave(y_train,y_valid)
    return x_train_valid, y_train_valid

def getKFoldSplit( xdata: np.ndarray, ydata: np.ndarray, nFolds: int, validFold: int, **kwargs ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    splitter = KFold( n_splits=nFolds, shuffle=kwargs.get("shuffle", False) )
    folds = list( splitter.split( xdata ) )
    train_indices, test_indices = folds[validFold]
    return xdata[train_indices], xdata[test_indices], ydata[train_indices], ydata[test_indices]

def getSplit( x_data: np.ndarray, y_data: np.ndarray, validation_fraction: float ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    NValidationElems = int( round( x_data.shape[0] * validation_fraction ) )
    NTrainingElems = x_data.shape[0] - NValidationElems
    x_train = x_data[:NTrainingElems]
    x_test =  x_data[NTrainingElems:]
    y_train = y_data[:NTrainingElems]
    y_test =  y_data[NTrainingElems:]
    return x_train, x_test, y_train, y_test