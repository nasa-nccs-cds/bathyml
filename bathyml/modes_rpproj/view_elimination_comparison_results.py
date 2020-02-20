from bathyml.common.data import *
import csv, functools, matplotlib.pyplot as plt
from geoproc.plot.bar import MultiBar
from multiprocessing import Pool

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
plotDir = os.path.join( outDir, "plots" )
resultsDir = os.path.expanduser("~/Dropbox/Tom/InnovationLab/results/FeatureImportance")

if not os.path.exists(plotDir): os.makedirs( plotDir )
training_percent = 80
n_estimators = 50
max_depth = 20
n_reduced_features = 4
n_trials = 100
plot_mse = False
plot_bars = True

results_file = os.path.join(outDir, f"fe.rf.variability-{n_reduced_features}-{training_percent}-{n_estimators}-{max_depth}.csv")
plot_file = os.path.join( resultsDir, f"fe.rf.variability-{n_reduced_features}-{training_percent}-{n_estimators}-{max_depth}.png" )

with open(results_file, "r") as csvfile:
    print(f"Read data from file {results_file}")
    csv_reader = csv.reader(csvfile)
    nf=[]
    mse_train=[]
    mse_test = []
    support = []
    for row in csv_reader:
        nf.append( int(row[0]) )
        mse_train.append( float(row[1]) )
        mse_test.append( float(row[2]) )
        support.append( [ int(row[x]) for x in range(3,len(row))] )

    if plot_mse:
        test_index = range( len(mse_train) )
        df = pd.DataFrame( dict( test_index=test_index, nFeatures=nf, train_mse=mse_train, test_mse=mse_test ) )

        fig, ax = plt.subplots()

        ax.plot('test_index', 'train_mse', data=df, color='green', linewidth=2)
        ax.plot('test_index', 'test_mse', data=df, color='blue', linewidth=2)
        ax.set_xlabel('test_index')
        ax.set_ylabel('MSE')
        ax.set_title(f'FE[{n_reduced_features}] MSE Plot, n_estimators={n_estimators}, max_depth={max_depth}')
        ax.grid(True)
        print( f"Saving plot to {plot_file}")
        fig.savefig( plot_file )
        ax.legend()
        plt.show()

    if plot_bars:
        band_names = [f"B-{iB}" for iB in range( 1, 36 )]
        barplots = MultiBar(f'FE[{n_reduced_features}] MSE Plot, n_estimators={n_estimators}, max_depth={max_depth}', band_names )
        bar_data = np.array( support ).sum( axis=0 )
        barplots.addPlot( f"Reduced Features, {n_trials} Trials ", bar_data )
        barplots.show()
