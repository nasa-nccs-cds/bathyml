from bathyml.common.data import *
import csv, matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from sklearn.feature_selection import RFE

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
n_inputs = 35
min_inputs = 5
elim_step = 2
make_plots = False

def write_array_data( outfile_path: str, data: List[np.ndarray]):
    with open(outfile_path, "w") as outfile:
        print(f"Write data to file {outfile_path}")
        csv_writer = csv.writer(outfile)
        for row in data:
            csv_writer.writerow(row)

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv" )
    x_data_norm: np.ndarray = EstimatorBase.normalize( x_data_raw[:,0:n_inputs] )

    modParms = dict(n_estimators=70, max_depth=20)
    estimator: EstimatorBase = EstimatorBase.new( "rf" )
    estimator.update_parameters( **modParms )
    print("Computing base fit")

    predictions = []
    feature_importance = []
    scores = []

    x_data_reduced = x_data_norm.copy()
    estimator.fit( x_data_reduced, y_data )
    prediction = estimator.predict(x_data_reduced)
    score = estimator.score(x_data_reduced, y_data)
    mse = mean_squared_error(prediction, y_data)
    scores.append(np.array([score, mse]))

    print( "Running RFE.fit")
    scores.append(np.array([score, mse]))
    reduced_inputs = n_inputs
    while True:
        reduced_inputs = reduced_inputs - elim_step
        if reduced_inputs < min_inputs: break
        rfe = RFE( estimator.instance, n_features_to_select=reduced_inputs, verbose=2 )
        rfe.fit( x_data_reduced, y_data )
        score = rfe.score( x_data_reduced, y_data )
        prediction = rfe.predict( x_data_reduced )
        mse = mean_squared_error( prediction, y_data )
        print( f"score = {score:.3f}, mse = {mse:.3f} ")
        predictions.append( rfe.predict( x_data_reduced ) )
        feature_importance.append( rfe.estimator_.feature_importances_ )
        scores.append( np.array( [ score, mse ] ) )
        x_data_reduced = rfe.transform( x_data_reduced )

    predictions_file = os.path.join(outDir, f"fe.rf.predictions.csv" )
    write_array_data( predictions_file, predictions )

    feature_importance_file = os.path.join(outDir, f"fe.rf.fi.csv" )
    write_array_data( feature_importance_file, feature_importance )

    scores_file = os.path.join(outDir, f"fe.rf.scores.csv" )
    write_array_data( scores_file, scores )