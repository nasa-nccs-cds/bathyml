from bathyml.common.data import *
import csv, functools, matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from sklearn.feature_selection import RFE
from multiprocessing import Pool

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
plotDir = os.path.join( outDir, "plots" )
if not os.path.exists(plotDir): os.makedirs( plotDir )
training_percent = 50
n_estimators = 35
max_depth = 10

results_file = os.path.join(outDir, f"fe.rf.results-{training_percent}.csv")
plot_file = os.path.join(plotDir, f"FE-RF-errors-{training_percent}-{n_estimators}-{max_depth}.png" )

with open(results_file, "r") as csvfile:
    print(f"Read data from file {results_file}")
    csv_reader = csv.reader(csvfile)
    nf=[]
    y_train=[]
    y_test = []
    for row in csv_reader:
        nf.append( int(row[0]) )
        y_train.append( float(row[1]) )
        y_test.append( float(row[2]) )
        features = [ int(row[x]) for x in range(3,len(row))]

    df = pd.DataFrame( dict( nFeatures=nf, train_mse=y_train, test_mse=y_test ) )

    fig, ax = plt.subplots()
    ax.plot('nFeatures', 'train_mse', data=df, color='green', linewidth=2)
    ax.plot('nFeatures', 'test_mse', data=df, color='blue', linewidth=2)
    ax.set_xlim( 10, 0 )
    ax.set_xlabel('N Features')
    ax.set_ylabel('MSE')
    ax.set_title(f'FE MSE Plot, n_estimators={n_estimators}, max_depth={max_depth}')
    ax.grid(True)
    print( f"Saving plot to {plot_file}")
    fig.savefig( plot_file )
    ax.legend()
    plt.show()