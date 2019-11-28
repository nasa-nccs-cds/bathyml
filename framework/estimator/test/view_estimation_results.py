from bathyml.common.data import *
from geoproc.plot.animation import ArrayListAnimation
import xarray as xa

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )

version = "T1"
image_name = "LC8_080010_20160709_stack_clip"
depth_map_path = os.path.join(outDir, f"depthMap.{image_name}.{version}.nc")

depth_map_dset = xa.open_dataset(depth_map_path)
depth_array = depth_map_dset["depth_map"]
animator = ArrayListAnimation([depth_array])
animator.show()

