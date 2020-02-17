import os, math, pickle, numpy as np
from geoproc.plot.bar import MultiBar

def norm( x: np.ndarray ): return x/x.mean()

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
plotDir = os.path.join( outDir, "plots" )
if not os.path.exists(plotDir): os.makedirs( plotDir )

modelType = "mlp"
n_inputs = 35
band_names = [f"B-{iB}" for iB in range(1, n_inputs + 1)]
nRuns = 5
bpvals_dataArray = None
method = "most_squares" # "common_weights"  "most_squares"

barplots = MultiBar(f"ANN Feature Importance: {method} Method", band_names )

for model_index in range(nRuns):
    init_weights_path = os.path.join(outDir, f"init_{modelType}_weights.T{model_index}.pkl")
    init_weights_file = open( init_weights_path, "rb")
    init_weights = pickle.load( init_weights_file )

    final_weights_path = os.path.join(outDir, f"final_{modelType}_weights.T{model_index}.pkl")
    final_weights_file = open(final_weights_path, "rb")
    final_weights = pickle.load( final_weights_file )

    if method == "common_weights":
        w0: np.ndarray = final_weights[0]
        w1: np.ndarray = final_weights[1]
        feature_importance = np.fabs( np.matmul( w0, w1 ).squeeze() )
        barplots.addPlot(f"M{model_index}", feature_importance )
    elif method == "most_squares":
        dw: np.ndarray = init_weights[0] - final_weights[0]
        feature_importance_mag = ( dw * dw ).sum( axis = 1 )
        feature_importance = feature_importance_mag / feature_importance_mag.sum( axis=0 )
        barplots.addPlot(f"M{model_index}", feature_importance)

barplots.addMeanPlot("Ave", write_to_file=os.path.join(outDir, f"fi.{method}.{modelType}.csv"))
barplots.save( os.path.join( plotDir, f"FI-{modelType}-weights-{method}.png" ) )
barplots.show()
