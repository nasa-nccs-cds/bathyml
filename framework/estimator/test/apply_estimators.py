from bathyml.common.data import *
from geoproc.plot.animation import ArrayListAnimation
from geoproc.xext.xgeo import XGeo
import xarray as xa

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
thisDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(thisDir))), "data")

version= "T1"
verbose = False
make_plots = False
show_plots = False
modelTypes = [ "mlp", "rf", "svr", "nnr" ]
modelType = modelTypes[1]
space_dims = ["y", "x"]
saveNetcdf = True
saveGeotiff = True

def mean_abs_error( x: np.ndarray, y: np.ndarray ):
    return np.mean( np.abs( x-y ), axis=0 )

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

def preprocess( x: xa.DataArray ):
    nodataval = x.nodatavals[0]
    xm: xa.DataArray = x.where( x != nodataval )
    mean = xm.mean( dim=space_dims, skipna=True )
    std = xm.std( dim=space_dims, skipna=True )
    results: xa.DataArray = ( xm - mean ) / std
    return results.stack( z=space_dims ).transpose().fillna(0.0).squeeze()

if __name__ == '__main__':

#    image_data_path = "/att/nobackup/maronne/lake/rasterStacks/080010/LC8_080010_20160709_stack_clip.tif"
    image_data_path = os.path.join( dataDir, "image", "LC8_080010_20160709_stack_clip.tif" )
    image_name = os.path.splitext(os.path.basename(image_data_path))[0]
    print( f"Reading data from file {image_data_path}")
    input_image: xa.DataArray = xa.open_rasterio( image_data_path, chunks=(35,1000,1000) )
    space_coords = { key: input_image.coords[key].values for key in space_dims }

    ml_input_data: xa.DataArray = preprocess( input_image )

    saved_model_path = os.path.join(outDir, f"model.{modelType}.{version}.pkl")
    filehandler = open(saved_model_path, "rb")
    estimator = pickle.load( filehandler )
    nodata_output = estimator.predict( np.zeros( [1, input_image.shape[0]] ) )[0]

    print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )
    ml_results: xa.DataArray = xa.apply_ufunc( estimator.predict, ml_input_data, input_core_dims=[['band']], dask="parallelized", output_dtypes=[np.float] )
    depth_map_data: np.ndarray = ml_results.values.reshape(input_image.shape[1:])
    result_map = xa.DataArray( depth_map_data, coords=space_coords, dims=space_dims, name="depth_map" )
    depth_map = result_map.where( result_map != nodata_output, 0.0 )

    if saveNetcdf:
        depth_map_path = os.path.join(outDir, f"depthMap.{image_name}.{version}.nc")
        print(f"Saving result to {depth_map_path}")
        depth_map.to_netcdf( depth_map_path )

    if saveGeotiff:
        depth_map_path = os.path.join(outDir, f"depthMap.{image_name}.{version}.tif")
        print(f"Saving result to {depth_map_path}")
        depth_map.xgeo.to_tif( depth_map_path )




