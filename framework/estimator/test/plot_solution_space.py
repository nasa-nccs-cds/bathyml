from bathyml.common.data import *
from geoproc.plot.animation import ArrayListAnimation
import random
from geoproc.xext.xgeo import XGeo
from geoproc.cluster.manager import ClusterManager
import joblib
import xarray as xa
import time

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
thisDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(thisDir))), "data")

version= "T3"
verbose = False
modelTypes = [ "mlp", "rf", "svr", "nnr" ]
modelType = modelTypes[1]
space_dims = ["y", "x"]
localTest = True
solution_point = [ 310,412,258,316,249,232,752,1202,981,1245,1336,1329,1597,1304,1655,1776,832,626,816,1036,1112,
                   1019,767,1225,1269,1362,803,604,965,788,748,563,899,734,932 ]
solution_depth = 4.14

if localTest:
    subset = False
    image_data_path = os.path.join(dataDir, "image", "LC8_080010_20160709_stack_clip.tif")
    cluster_parameters = { "log.scheduler.metrics": False, 'type': 'local' }
else:
    subset = False
    image_data_path = "/att/nobackup/maronne/lake/rasterStacks/080010/LC8_080010_20160709_stack_clip.tif"
    cluster_parameters = { "log.scheduler.metrics": False, 'type': 'slurm' }

def generate_bands( Blue: float, Green: float, Red: float, NIR: float, SWIR1: float, SWIR2: float ):
    result = [Blue, Green, Red, NIR, SWIR1, SWIR2,
              Blue/Green, Blue/Red, Blue/NIR, Blue/SWIR1, Blue/SWIR2,
              Green/Blue, Green/Red, Green/NIR, Green/SWIR1, Green/SWIR2,
              Red/Blue, Red/Green, Red/NIR, Red/SWIR1, Red/SWIR2,
              NIR/Blue, NIR/Green, NIR/Red, NIR/SWIR1, NIR/SWIR2,
              SWIR1/Blue, SWIR1/Green, SWIR1/Red, SWIR1/NIR,
              SWIR2/Blue, SWIR2/Green, SWIR2/Red, SWIR2/NIR, SWIR2/SWIR1]
    return np.array( result )

def get_scaling( x: np.ndarray ):
    mean = np.mean( x, axis=0 )
    std = np.std( x, axis=0 )
    return mean, std

def rescale( x: np.ndarray, mean0: np.ndarray, std0: np.ndarray ):
    mean1 = np.mean( x, axis=0 )
    std1 = np.std( x, axis=0 )
    mean = np.hstack( [mean0[0:6], mean1[6:]] )
    std = np.hstack( [std0[0:6], std1[6:]] )
    return ( x - mean )/std

def generate_solution_space( b01_shape: List[int], b01_resolution: float,  b2Value: int = solution_point[2],  b3Value: int = solution_point[3] ) -> np.ndarray:
    results = []
    for i0 in range( 0, b01_shape[0] ):
        b0 = solution_point[0] + b01_resolution * i0
        for i1 in range( 0, b01_shape[1] ):
            b1 = solution_point[1] + b01_resolution * i1
            results.append( generate_bands( b0, b1, b2Value, b3Value, solution_point[4], solution_point[5] ) )
    return np.vstack( results )

if __name__ == '__main__':

    solution_space_shape = [ 64, 64 ]
    b2Res = 0.2
    b01Res = 1.0/solution_space_shape[0]
    nSlices = 3

    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv" )
    input_norm, input_std = get_scaling( x_data_raw )

    saved_model_path = os.path.join(outDir, f"model.{modelType}.{version}.pkl")
    filehandler = open(saved_model_path, "rb")
    estimator = pickle.load(filehandler)

    print( f"Executing {modelType} estimator: {saved_model_path}, parameters: { list(estimator.instance_parameters.items()) }" )
    solution_slices = []
    for b2Index in range( 0, nSlices ):
        b2Val = solution_point[2] + b2Index * b2Res
        raw_input_data: np.ndarray = generate_solution_space( solution_space_shape, b01Res, b2Val )
        ml_input_data = rescale( raw_input_data, input_norm, input_std )
        ml_results: np.ndarray = estimator.predict( ml_input_data )
        mse = np.abs( ml_results - solution_depth ).reshape( solution_space_shape )
        solution_slices.append( xa.DataArray( mse, dims=["y","x"], name=f"band2={b2Val}" )  )

    anim = ArrayListAnimation( solution_slices )
    anim.show()
