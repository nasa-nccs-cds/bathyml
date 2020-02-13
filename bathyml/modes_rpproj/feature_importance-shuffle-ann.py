from bathyml.common.data import *
import matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
from geoproc.plot.bar import MultiBar

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
nVersions = 8
n_inputs = 35
verbose = False
modelType =  "mlp" #[ "mlp", "rf", "svr", "nnr" ]
make_plots = True

parameters = dict(
    mlp=dict( max_iter=500, learning_rate="constant", solver="adam", early_stopping=False ),
    rf=dict(n_estimators=70, max_depth=20),
    svr=dict(C=5.0, gamma=0.5, cache_size=2000 ),
    nnr=dict( n_neighbors=3, weights='distance', algorithm = 'kd_tree', leaf_size=30, metric="euclidean", n_jobs=8 ),
)

def mean_abs_error( x: np.ndarray, y: np.ndarray ):
    return np.mean( np.abs( x-y ), axis=0 )

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

def shuffle_data(input_data, training_data ): #  -> ( shuffled_input_data, shuffled_training_data):
    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    shuffled_input_data, shuffled_training_data = np.copy(input_data), np.copy(training_data)
    shuffled_input_data[:] = input_data[indices]
    shuffled_training_data[:] = training_data[indices]
    return ( shuffled_input_data, shuffled_training_data)


def shuffle_feature( input_data: np.ndarray, iFeature: int ) -> np.ndarray:
    features = np.split( input_data, input_data.shape[1], axis=1 )
    shuffled_feature = np.copy( features[ iFeature ] )
    np.random.shuffle( shuffled_feature )
    features[iFeature] = shuffled_feature
    result = np.stack( features, axis=1 ).squeeze()
    return result

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv" )
    x_data_norm = EstimatorBase.normalize( x_data_raw[:,0:n_inputs], 1 )

    band_names = ['b1_LC8_075', 'b2_LC8_075', 'b3_LC8_075', 'b4_LC8_075', 'b5_LC8_075', 'b6_LC8_075', 'b7_LC8_075', 'b8_LC8_075', 'b9_LC8_075', 'b10_LC8_07',
                  'b11_LC8_07', 'b12_LC8_07', 'b13_LC8_07', 'b14_LC8_07', 'b15_LC8_07', 'b16_LC8_07', 'b17_LC8_07', 'b18_LC8_07', 'b19_LC8_07', 'b20_LC8_07',
                  'b21_LC8_07', 'b22_LC8_07', 'b23_LC8_07', 'b24_LC8_07', 'b25_LC8_07', 'b26_LC8_07', 'b27_LC8_07', 'b28_LC8_07', 'b29_LC8_07', 'b30_LC8_07',
                  'b31_LC8_07', 'b32_LC8_07', 'b33_LC8_07', 'b34_LC8_07', 'b35_LC8_07']

    barplots = MultiBar("MLP Feature Importance", band_names)
    feature_importances = []
    for iVersion in range(nVersions):
        saved_model_path = os.path.join(outDir, f"model.{modelType}.T{iVersion}.pkl")
        print( f"Loading estimator from {saved_model_path}" )
        filehandler = open(saved_model_path, "rb")
        estimator = pickle.load(filehandler)

        baseline_prediction = estimator.predict( x_data_norm )
        baseline_mse =  mean_squared_error( y_data, baseline_prediction )

        print(f" Baseline MSE: {baseline_mse}")
        feature_importance = []

        for iFeature in range( x_data_norm.shape[1] ):
            x_data_norm_shuffled = shuffle_feature( x_data_norm, iFeature )
            train_prediction = estimator.predict( x_data_norm_shuffled )
            shuffled_mse =  mean_squared_error( y_data, train_prediction )

            del_mse = shuffled_mse - baseline_mse
#            print( f" Shuffled MSE-{iFeature}: {shuffled_mse}, del = {del_mse}" )
            feature_importance.append( del_mse )

        np_feature_importance = np.array( feature_importance )
        barplots.addPlot( f"M-{iVersion}", np_feature_importance )

    barplots.addMeanPlot(f"M-Ave" )
    barplots.show()
