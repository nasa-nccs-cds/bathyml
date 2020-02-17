from bathyml.common.data import *
from framework.estimator.base import EstimatorBase
from sklearn.ensemble import RandomForestRegressor
from geoproc.plot.bar import MultiBar

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
thisDir = os.path.dirname(os.path.abspath(__file__))
dataDir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(thisDir))), "data")
plotDir = os.path.join( outDir, "plots" )
if not os.path.exists(plotDir): os.makedirs( plotDir )

band_names = [ 'b1_LC8_075', 'b2_LC8_075', 'b3_LC8_075', 'b4_LC8_075', 'b5_LC8_075', 'b6_LC8_075', 'b7_LC8_075', 'b8_LC8_075', 'b9_LC8_075', 'b10_LC8_07',
               'b11_LC8_07', 'b12_LC8_07', 'b13_LC8_07', 'b14_LC8_07', 'b15_LC8_07', 'b16_LC8_07', 'b17_LC8_07', 'b18_LC8_07', 'b19_LC8_07', 'b20_LC8_07',
               'b21_LC8_07', 'b22_LC8_07', 'b23_LC8_07', 'b24_LC8_07', 'b25_LC8_07', 'b26_LC8_07', 'b27_LC8_07', 'b28_LC8_07', 'b29_LC8_07', 'b30_LC8_07',
               'b31_LC8_07', 'b32_LC8_07', 'b33_LC8_07', 'b34_LC8_07', 'b35_LC8_07' ]

nVersions = 5
n_inputs = 35
space_dims = ["y", "x"]
localTest = True

if __name__ == '__main__':

    barplots = MultiBar( "RF Feature Importance: RFR Method", band_names[:n_inputs] )

    for iVersion in range(nVersions):
        saved_model_path = os.path.join(outDir, f"model.rf.T{iVersion}.pkl")
        filehandler = open(saved_model_path, "rb")
        estimator: EstimatorBase = pickle.load(filehandler)
        instance: RandomForestRegressor = estimator.instance
        barplots.addPlot( f"M{iVersion}", instance.feature_importances_ )

    barplots.addMeanPlot( "Ave", write_to_file=os.path.join(outDir, f"fi.internal.rf.csv") )
    barplots.save(os.path.join(plotDir, f"FI-rf-internal.png"))
    barplots.show()
