from bathyml.common.data import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xarray as xa
from framework.estimator.base import EstimatorBase
import pandas as pd

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
thisDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(thisDir))), "data")

version= 2
verbose = False
make_plots = False
show_plots = False
modelTypes = [ "mlp", "rf", "svr", "nnr" ]
modelType = modelTypes[1]

parameters = dict(
    mlp=dict( max_iter=150, learning_rate="constant", solver="adam", early_stopping=False ),
    rf=dict(n_estimators=70, max_depth=20),
    svr=dict(C=5.0, gamma=0.5),
    nnr=dict( n_neighbors=5, weights='distance' ),
)

def mean_abs_error( x: np.ndarray, y: np.ndarray ):
    return np.mean( np.abs( x-y ), axis=0 )

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

if __name__ == '__main__':
    print("Reading Data")
#    image_data_path = "/att/nobackup/maronne/lake/rasterStacks/080010/LC8_080010_20160709_stack_clip.tif"
    image_data_path = os.path.join( dataDir, "image", "LC8_080010_20160709_stack_clip.tif" )
    image_data = xa.open_rasterio( image_data_path )
    shp = image_data.shape
    masked_image_data: np.ma.MaskedArray = np.ma.masked_values( image_data.values, image_data[0,0,0] ).reshape( shp[0], shp[1]*shp[2] ).transpose()
    ml_input_data: np.ma.MaskedArray = np.ma.masked_values( preprocessing.scale( masked_image_data ), float('nan') )

    saved_model_path = os.path.join(outDir, f"model.{modelType}.{version}.pkl")
    filehandler = open(saved_model_path, "rb")
    estimator = pickle.load( filehandler )

    print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )
    estimation_data = estimator.predict( ml_input_data )
    depth_map = estimation_data.reshape( shp[1], shp[2] )
    xa_depth_map = xa.DataArray( depth_map, coords = image_data.coords, dims = image_data.dims[1:] )
    xa_depth_map.plot()
    plt.show()



