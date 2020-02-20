from bathyml.common.data import *
import csv, functools, matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from sklearn.feature_selection import RFE
from multiprocessing import Pool

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
    rfe = RFE(estimator, n_features_to_select=nFeatures, verbose=2, step = step )
    rfe.fit(x_data_train, y_data_train)
    prediction = rfe.predict(x_data_train)
    test_mse = mean_squared_error(prediction, y_data_train )
    generalization = rfe.predict(x_data_test)
    gen_mse = mean_squared_error( generalization, y_data_test )
    return [ nFeatures, f"{test_mse:.3f}", f"{gen_mse:.3f}" ] + [ int(x) for x in rfe.support_ ]


if __name__ == '__main__':
    print("Reading Data")
    training_fraction = 0.50
    reduction_step = 1
    n_estimators = 35
    max_depth = 10
    nproc = 4

    pts_data, x_data_raw, y_data_raw = read_csv_data( "pts_merged_final.csv" )
    n_total_samples = x_data_raw.shape[0]
    n_training_samples = int(n_total_samples*training_fraction)
    nFeaturesList = list( range(1,11) )

    x_data_train: np.ndarray = EstimatorBase.normalize( x_data_raw[:n_training_samples] )
    y_data_train = y_data_raw[:n_training_samples]
    x_data_test: np.ndarray = EstimatorBase.normalize( x_data_raw[n_training_samples:] )
    y_data_test = y_data_raw[n_training_samples:]

    modParms = dict( n_estimators=n_estimators, max_depth=10 )
    estimatorBase: EstimatorBase = EstimatorBase.new( "rf" )
    estimatorBase.update_parameters( **modParms )

    print("Computing feature reductions")
    run_feature_reduction = functools.partial( feature_reduction, estimatorBase.instance, x_data_train,
                                               y_data_train, x_data_test, y_data_test, reduction_step )
    with Pool(processes=nproc) as pool:
        results = pool.map( run_feature_reduction, nFeaturesList )

    results_file = os.path.join(outDir, f"fe.rf.results-{int(training_fraction*100)}-{n_estimators}-{max_depth}.csv" )
    write_array_data( results_file, results )