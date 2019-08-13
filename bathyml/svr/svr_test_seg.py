from bathyml.common.data import *
import matplotlib.pyplot as plt
from sklearn import svm
from time import time
from datetime import datetime

svmArgs = dict(
    C=1.0,
    cache_size=400,
    coef0=0.0,
    degree=3,
    epsilon=0.1,
    gamma=0.1,
    kernel='rbf',
    max_iter=-1,
    shrinking=True,
    tol=0.001,
    verbose=False )

validation_fraction = 0.2
network_label = f"SVM-{svmArgs['kernel']}-{svmArgs['C']:.2f}-{svmArgs['gamma']:.2f}"
nRuns = 1

if __name__ == '__main__':
    print("Reading Data")
    x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    x_valid: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )

    x_train_valid, y_train_valid = getTrainingData( x_train, y_train, x_valid, y_valid )
    nValidSamples = int( round( x_train_valid.shape[0] * validation_fraction ) )
    nTrainSamples = x_train_valid.shape[0] - nValidSamples
    input_dim = x_train.shape[1]

    ens_min_loss = sys.float_info.max
    best_model = None
    best_prediction_validation = None
    best_prediction_training = None

    for c in np.logspace(-2, 2, 5):
        for g in np.logspace(-2, 2, 5):
            svmArgs['C'] = c
            svmArgs['gamma'] = g
            model = svm.SVR( **svmArgs )
            print( f"Fitting Model, c={c}, gamma={g}" )
            model.fit( x_train_valid[:nTrainSamples], y_train_valid[:nTrainSamples] )
            print("Creating Prediction")
            prediction_validation = model.predict( x_train_valid[nTrainSamples:] )
            train_loss = abs(prediction_validation - y_train_valid[nTrainSamples:]).mean()

            if train_loss < ens_min_loss:
                ens_min_loss = train_loss
                best_model = model
                best_prediction_validation = prediction_validation
                best_prediction_training = model.predict( x_train_valid[:nTrainSamples] )

    print( f"Plotting results (loss={ens_min_loss})")
    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    ax0 = plt.subplot("211")
    ax0.set_title( f"{network_label} Training Data")
    xaxis = range(best_prediction_training.shape[0])
    ax0.plot(xaxis, y_train_valid[:nTrainSamples], "b--", label="validation data")
    ax0.plot(xaxis, best_prediction_training, "r--", label="prediction")
    ax0.legend()

    ax1 = plt.subplot("212")
    ax1.set_title( f"{network_label} Validation Data MAE = {ens_min_loss:.2f} " )
    xaxis = range(best_prediction_validation.shape[0])
    ax1.plot( xaxis, y_train_valid[nTrainSamples:], "b--", label="training data")
    ax1.plot( xaxis, best_prediction_validation, "r--", label="prediction")
    ax1.legend()

    saved_results_path = os.path.join(outDir, f"results_{network_label}__{datetime.now().strftime('%m-%d-%H.%M.%S')}")
    filehandler = open(saved_results_path, "wb")
    pickle.dump( ( y_train_valid.tolist(), best_prediction_training.tolist(), best_prediction_validation.tolist() ) , filehandler)
    print(f"Saved results to file {saved_results_path}")

    plt.tight_layout()
    plt.show()