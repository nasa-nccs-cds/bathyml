from geoproc.data.shapefiles import ShapefileManager
import os
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname(HERE) ), "data" )
outDir = os.path.join(DATA, "results")
if not os.path.exists(outDir): os.makedirs( outDir )
SHP =os.path.join( DATA, "gis", "mergedJonesHumBird", "hum_jones_pts_merge.shp" )
CSV =os.path.join( DATA, "csv", "mergedJonesHumBird.csv"  )

shpManager = ShapefileManager()
gdShape: gpd.GeoDataFrame = shpManager.read( SHP )
gdShape.to_csv( CSV, index=False, header=False )
print( f"Converted shaplefile '{SHP}' to csv '{CSV}'." )
