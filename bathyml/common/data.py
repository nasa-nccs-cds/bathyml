from typing import List, Optional, Tuple, Dict, Any
import os, copy, sys, numpy as np, pickle, math
from sklearn.model_selection import KFold
import pandas as pd

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

class IterativeTable:

    def __init__( self, cols = List[str] ):
        self.rows = []
        self.cols = cols
        self.cached_table = None
        self.default_index = 0

    def add_row( self, index=None, data: List = None ):
        self.cached_table = None
        if index is None:
            index = self.default_index
            self.default_index = self.default_index + 1
        self.rows.append( pd.DataFrame.from_dict( { index: data }, orient="index", columns=self.cols )  )

    def get_table(self) -> pd.DataFrame:
        if self.cached_table == None:
            self.cached_table = pd.concat( self.rows )
        return  self.cached_table

    def get_sums(self) -> pd.DataFrame:
        return pd.concat( self.rows ).sum( axis=0 )

    def to_csv( self, filePath, **kwargs ):
        self.get_table().to_csv( filePath, **kwargs )


def read_csv_data( fileName: str, **kwargs ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    import csv
    file_path: str = os.path.join( ddir, fileName )
    with open(file_path) as csvfile:
        csvData = csv.reader( csvfile, delimiter=',' )
        headers = None
        ydata = []
        xdata = []
        fids = []
        current_fid = -1
        object_index = 0
        for index,row in enumerate(csvData):
            if index == 0: headers = row
            else:
                fid = int(row[0])
                if current_fid >= fid:
                    object_index = object_index + 1
                current_fid = fid
                ydata.append( float(row[1]) )
                xdata.append( [ float(r) for r in row[3:] ] )
                fids.append( [object_index, fid] )
        print( f"Reading csv datra from {file_path}, headers = {headers}" )
        np_xdata = np.array( xdata )
        np_ydata = np.array( ydata )
        np_ptdata = np.array( fids )
        return np_ptdata, np_xdata, np_ydata

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

def getKFoldSplit( ptsData: np.ndarray, xData: np.ndarray, ydata: np.ndarray, nFolds: int, validFold: int, **kwargs ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if nFolds < 2: return ptsData, np.empty([0,0]), xData, np.empty(0), ydata, np.empty([0,0])
    splitter = KFold( n_splits=nFolds, shuffle=kwargs.get("shuffle", False) )
    folds = list( splitter.split( xData ) )
    train_indices, test_indices = folds[validFold]
    return  ptsData[train_indices], ptsData[test_indices], xData[train_indices], xData[test_indices], ydata[train_indices], ydata[test_indices]

def getSplit( x_data: np.ndarray, y_data: np.ndarray, validation_fraction: float ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if validation_fraction == 0.0: return x_data, np.empty([0,0]), y_data, np.empty([0,0])
    NValidationElems = int( round( x_data.shape[0] * validation_fraction ) )
    NTrainingElems = x_data.shape[0] - NValidationElems
    x_train = x_data[:NTrainingElems]
    x_test =  x_data[NTrainingElems:]
    y_train = y_data[:NTrainingElems]
    y_test =  y_data[NTrainingElems:]
    return x_train, x_test, y_train, y_test