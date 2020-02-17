from bathyml.common.data import *
import matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from geoproc.plot.bar import MultiBar

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
plotDir = os.path.join( outDir, "plots" )
if not os.path.exists(plotDir): os.makedirs( plotDir )
nVersions = 5
n_inputs = 35
verbose = False
modelTypes = [ "rf", "mlp", "svr", "nnr" ]
show_plots = True
#methods = [ "shuffle", "common_weights", "most_squares" ]
#comp_type = "ANN_FI_Methods" # ""ShuffleMethod"
#methods = [ "shuffle", "internal" ]
#comp_type = "RF_FI_Methods" # ""ShuffleMethod"
methods = [ "shuffle" ]
comp_type = "ShuffleMethod" # ""ShuffleMethod"

if __name__ == '__main__':

    band_names = [ f"B-{iB}" for iB in range(1,n_inputs+1)]
    barplots = MultiBar(f"Feature Importance Comparison: {comp_type}", band_names)

    for modelType in modelTypes:
        for method in methods:

            saved_data_path = os.path.join(outDir, f"fi.{method}.{modelType}.csv")
            print( f"Loading data from {saved_data_path}" )
            barplots.load_plot_data( saved_data_path, f"{modelType}")

    barplots.addMeanPlot(f"Ave" )
    barplots.save( os.path.join( plotDir, f"FIC-{comp_type}.png"  ) )
    if show_plots: barplots.show()
