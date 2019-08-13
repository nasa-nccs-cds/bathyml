from typing import List, Optional, Tuple, Dict, Any
import os, copy, sys, numpy as np, pickle, math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname(HERE) ), "data" )
outDir = os.path.join(DATA, "results")
if not os.path.exists(outDir): os.makedirs( outDir )
ddir = os.path.join(DATA, "csv")
tb_log_dir=f"{ddir}/logs/tb"
nBands = 21

def read_csv_data( fileName: str, nBands: int = 0 ) -> np.ndarray:
    file_path: str = os.path.join( ddir, fileName )
    raw_data_array: np.ndarray = np.loadtxt( file_path, delimiter=',')
    if (nBands > 0): raw_data_array = raw_data_array[:,:nBands]
    return raw_data_array

def normalize( array: np.ndarray, scale = 1.5 ):
    ave = array.mean( axis=0 )
    std = array.std( axis=0 )
    return (array-ave)/(scale*std)

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

def getTrainingData( x_train, y_train, x_valid, y_valid ) -> Tuple[np.ndarray,np.ndarray]:
    x_train_valid = normalize( interleave(x_train,x_valid) )
    y_train_valid = normalize( interleave(y_train,y_valid) )
    return x_train_valid, y_train_valid