from bathyml.common.data import *
import xarray as xa
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from typing import Dict, List, Tuple, Union
import os, time, sys


scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )

version = "T1"
image_name = "LC8_080010_20160709_stack_clip"
depth_map_path = os.path.join(outDir, f"depthMap.{image_name}.{version}.nc")

depth_map_dset = xa.open_dataset(depth_map_path)
depth_array = depth_map_dset["depth_map"]

figure, axes = plt.subplots()
im: Image = axes.imshow(depth_array.values, cmap="jet" )
plt.show()




