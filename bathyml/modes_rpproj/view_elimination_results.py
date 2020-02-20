from bathyml.common.data import *
import csv, matplotlib.pyplot as plt
import pandas as pd
from geoproc.plot.bar import MultiBar

n_inputs = 35
show_fi = False
show_mse = True

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
plotDir = os.path.join( outDir, "plots" )
if not os.path.exists(plotDir): os.makedirs( plotDir )


if show_fi:
    feature_importance_file = os.path.join(outDir, f"fe.rf.fi.csv")
    band_names = [f"B-{iB}" for iB in range(1, n_inputs + 1)]

    barplots = MultiBar(f"RF Feature Importance: Progressive Elimination", band_names)

    with open(feature_importance_file, "r") as csvfile:
        print(f"Read data from file {feature_importance_file}")
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            plot_data: np.array = np.array([float(x) for x in row])
            nFeatures = np.count_nonzero(plot_data)
            barplots.addPlot(f"F-{nFeatures}", plot_data)

    barplots.save(os.path.join(plotDir, f"FE-RF-elimination.png"))
    barplots.show()


if show_mse:
    scores_file = os.path.join(outDir, f"fe.rf.scores.csv" )
    nFeatures = list( range( 35, 0, -2 ) )

    with open(scores_file, "r") as csvfile:
        print(f"Read data from file {scores_file}")
        csv_reader = csv.reader(csvfile)
        train_mse, test_mse = [], []
        for row in csv_reader:
            train_mse.append( float(row[1]) )
            test_mse.append( float(row[2]) )

        df = pd.DataFrame( dict( nFeatures=nFeatures, train_mse=train_mse, test_mse=test_mse) )

        fig, ax = plt.subplots()
        ax.plot('nFeatures', 'train_mse', data=df, color='green', linewidth=2)
        ax.plot('nFeatures', 'test_mse', data=df, color='blue', linewidth=2)
        ax.set_xlim( 36, 0 )
        ax.set_xlabel('N Features')
        ax.set_ylabel('MSE')
        ax.set_title('Error Plot for Feature Elimination')
        ax.grid(True)
        fig.savefig( os.path.join(plotDir, f"FE-RF-errors.png" ) )
        ax.legend()
        plt.show()

