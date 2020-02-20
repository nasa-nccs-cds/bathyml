from bathyml.common.data import *
import csv, functools
from framework.estimator.base import EstimatorBase
from sklearn.feature_selection import RFE
from multiprocessing import Pool
from bathyml.common.data import getKFoldSplit

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )

def write_array_data( outfile_path: str, data: List[np.ndarray]):
    with open(outfile_path, "w") as outfile:
        print(f"Write data to file {outfile_path}")
        csv_writer = csv.writer(outfile)
        for row in data:
            csv_writer.writerow(row)

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

def feature_reduction( estimator, x_data_train, y_data_train, x_data_test, y_data_test, step, nFeatures ):
    rfe = RFE(estimator.instance, n_features_to_select=nFeatures, verbose=2, step = step )
    rfe.fit( *estimator.shuffle_data( x_data_train, y_data_train ) )
    prediction = rfe.predict(x_data_train)
    test_mse = mean_squared_error(prediction, y_data_train )
    generalization = rfe.predict(x_data_test)
    gen_mse = mean_squared_error( generalization, y_data_test )
    return [ nFeatures, f"{test_mse:.3f}", f"{gen_mse:.3f}" ] + [ int(x) for x in rfe.support_ ]

if __name__ == '__main__':
    print("Reading Data")
    n_folds = 5
    reduction_step = 1
    nTrials = 100
    n_estimators = 50
    max_depth = 20
    nFeatures = 4
    nproc = 8

    pts_data, x_data_raw, y_data_raw = read_csv_data( "pts_merged_final.csv" )
    x_data_norm: np.ndarray = EstimatorBase.normalize(x_data_raw)
    nFeaturesList = [ nFeatures ]*nTrials

    for iFold in range(n_folds):
        pts_train, pts_valid, x_data_train, x_data_test, y_data_train, y_data_test = getKFoldSplit( pts_data, x_data_norm, y_data_raw, n_folds, iFold )

        modParms = dict( n_estimators=n_estimators, max_depth=10 )
        estimator: EstimatorBase = EstimatorBase.new( "rf" )
        estimator.update_parameters( **modParms )

        print("Computing feature reductions")
        run_feature_reduction = functools.partial(feature_reduction, estimator, x_data_train, y_data_train, x_data_test, y_data_test, reduction_step)
        if nproc > 1:
            with Pool(processes=nproc) as pool:
                results = pool.map( run_feature_reduction, nFeaturesList )
        else:
            results = [ run_feature_reduction(nF) for nF in nFeaturesList ]

        results_file = os.path.join(outDir, f"fe.rf.variability-{nFeatures}-{n_folds}-{n_estimators}-{max_depth}-{iFold}.csv" )
        write_array_data( results_file, results )