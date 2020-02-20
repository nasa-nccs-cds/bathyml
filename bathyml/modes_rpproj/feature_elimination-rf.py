from bathyml.common.data import *
import csv, matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from sklearn.feature_selection import RFE

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )

training_fraction = 0.666
min_inputs = 1
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

def reduce_band_names( band_names, mask ):
    reduced_band_names = []
    for bname,mval in zip( band_names, mask ):
        if mval: reduced_band_names.append( bname )
    return reduced_band_names

def pad_fe( band_names, reduced_band_names, feature_importances ) -> np.array:
    padded_fe = []
    iRBN = 0
    for iB in range( len( band_names ) ):
        if (iRBN < len(reduced_band_names)) and (band_names[iB] == reduced_band_names[iRBN]):
            padded_fe.append( feature_importances[iRBN] )
            iRBN = iRBN + 1
        else:
            padded_fe.append( 0.0 )
    return np.array( padded_fe )

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data_raw = read_csv_data( "pts_merged_final.csv" )
    n_inputs = x_data_raw.shape[1]
    band_names = [f"B-{iB}" for iB in range(1, n_inputs + 1)]
    n_total_samples = x_data_raw.shape[0]
    n_training_samples = int(n_total_samples*training_fraction)

    x_data_train: np.ndarray = EstimatorBase.normalize( x_data_raw[:n_training_samples] )
    y_data_train = y_data_raw[:n_training_samples]
    x_data_test: np.ndarray = EstimatorBase.normalize( x_data_raw[n_training_samples:] )
    y_data_test = y_data_raw[n_training_samples:]

    modParms = dict(n_estimators=70, max_depth=20)
    estimator: EstimatorBase = EstimatorBase.new( "rf" )
    estimator.update_parameters( **modParms )
    print("Computing base fit")

    predictions = []
    feature_importance = []
    scores = []

    train_data_reduced = x_data_train.copy()
    test_data_reduced = x_data_test.copy()
    estimator.fit( train_data_reduced, y_data_train )
    prediction = estimator.predict(train_data_reduced)
    score = estimator.score(train_data_reduced, y_data_train )
    train_mse = mean_squared_error(prediction, y_data_train )
    generalization = estimator.predict(test_data_reduced)
    gen_mse = mean_squared_error( generalization, y_data_test )
    scores.append(np.array([ score, train_mse, gen_mse ]))

    print( "Running RFE.fit")
    reduced_inputs = n_inputs
    reduced_band_names = np.array( band_names )
    while True:
        reduced_inputs = reduced_inputs - elim_step
        if reduced_inputs < min_inputs: break
        rfe = RFE( estimator.instance, n_features_to_select=reduced_inputs, verbose=2 )
        rfe.fit( train_data_reduced, y_data_train )
        reduced_band_names = reduce_band_names( reduced_band_names, rfe.support_ )
        score = rfe.score( train_data_reduced, y_data_train )
        prediction = rfe.predict( train_data_reduced )
        test_mse = mean_squared_error( prediction, y_data_train )
        generalization = rfe.predict( test_data_reduced )
        gen_mse = mean_squared_error( generalization, y_data_test )
        print( f"score = {score:.3f}, mse = {test_mse:.3f}, gen_mse = {gen_mse:.3f} ")
        predictions.append( rfe.predict( train_data_reduced ) )
        feature_importance.append( pad_fe( band_names, reduced_band_names, rfe.estimator_.feature_importances_) )
        scores.append( np.array( [ score, test_mse, gen_mse ] ) )
        train_data_reduced = rfe.transform( train_data_reduced )
        test_data_reduced = rfe.transform( test_data_reduced )

    predictions_file = os.path.join(outDir, f"fe.rf.predictions.csv" )
    write_array_data( predictions_file, predictions )

    feature_importance_file = os.path.join(outDir, f"fe.rf.fi.csv" )
    write_array_data( feature_importance_file, feature_importance )

    scores_file = os.path.join(outDir, f"fe.rf.scores.csv" )
    write_array_data( scores_file, scores )