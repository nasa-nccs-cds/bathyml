from bathyml.common.data import *
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn import preprocessing
from time import time
from datetime import datetime

krrArgs: Dict = dict(
    kernel_params=None,
    coef0=1,
    degree=3,
    alpha=1.0,
    gamma=2.0,
    kernel='rbf' )

validation_fraction = 0.2
pca_components = 0 # 14
whiten = False

if __name__ == '__main__':
    print("Reading Data")
    x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    x_valid: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )

    x_data, y_data = getTrainingData( x_train, y_train, x_valid, y_valid )

    if pca_components > 0:
        pca = PCA( n_components = pca_components, whiten=whiten )
        x_data_norm = pca.fit( x_data ).transform( x_data )
        if not whiten: x_data_norm = preprocessing.scale( x_data_norm )
        print(f'PCA: explained variance ratio ({pca_components} components): {pca.explained_variance_ratio_}' )
    else:
        x_data_norm = preprocessing.scale( x_data )

    input_dim = x_train.shape[1]
    NValidationElems = int( round( x_data.shape[0] * validation_fraction ) )
    NTrainingElems = x_data.shape[0] - NValidationElems

    x_train = x_data_norm[:NTrainingElems]
    x_test =  x_data_norm[NTrainingElems:]
    y_train = y_data[:NTrainingElems]
    y_test =  y_data[NTrainingElems:]
    ens_min_loss = sys.float_info.max
    best_model = None
    best_prediction_validation = None

    alpha_range = [ 1.0 ]
    gamma_range = [ 0.1 ]
    results = {}
    best_alpha = None
    best_gamma = None

    for a in alpha_range:
        for g in gamma_range:
            krrArgs['alpha'] = a
            krrArgs[ 'gamma' ] = g
            model = KernelRidge( **krrArgs )
            model.fit( x_train, y_train )
            prediction_validation = model.predict( x_test )
            diff = prediction_validation - y_test
            train_loss = math.sqrt( (diff*diff).mean() )
            print(f"Fitting Model, kernel={krrArgs['kernel']}, A={a}, gamma={g}, Validation Loss = {train_loss}" )

            if train_loss < ens_min_loss:
                ens_min_loss = train_loss
                best_model = model
                best_prediction_validation = prediction_validation
                best_alpha = a
                best_gamma = g

    best_prediction_training = best_model.predict( x_train ) 
    model_label = "-".join(["KRR", krrArgs['kernel'], str(best_alpha), str(best_gamma)])
    ref_loss =  math.sqrt( (best_prediction_validation*best_prediction_validation).mean() )
    print( f"Plotting results: loss={ens_min_loss}, model={model_label}, ref loss = {ref_loss}")
    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    diff = y_train - best_prediction_training
    mse = math.sqrt((diff * diff).mean())
    ax0 = plt.subplot("211")
    ax0.set_title( f"{model_label} Training Data  MSE = {mse:.2f}")
    xaxis = range(best_prediction_training.shape[0])
    ax0.plot(xaxis, y_train, "b--", label="validation data")
    ax0.plot(xaxis, best_prediction_training, "r--", label="prediction")
    ax0.legend()

    diff = y_test - best_prediction_validation
    mse = math.sqrt((diff * diff).mean())
    ref_mse = math.sqrt( (y_test*y_test).mean() )
    print( f" REF MSE = {ref_mse} ymax = {y_test.max()}")
    ax1 = plt.subplot("212")
    ax1.set_title( f"{model_label} Validation Data MSE = {mse:.2f} " )
    xaxis = range(best_prediction_validation.shape[0])
    ax1.plot( xaxis, y_test, "b--", label="training data")
    ax1.plot( xaxis, best_prediction_validation, "r--", label="prediction")
    ax1.legend()

    saved_results_path = os.path.join(outDir, f"results_{model_label}__{datetime.now().strftime('%m-%d-%H.%M.%S')}")
    filehandler = open(saved_results_path, "wb")
    pickle.dump( ( best_prediction_training.tolist(), best_prediction_validation.tolist() ) , filehandler)
    print(f"Saved results to file {saved_results_path}")

    plt.tight_layout()
    plt.show()