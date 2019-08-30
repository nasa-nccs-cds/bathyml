from bathyml.common.data import *
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import preprocessing, neighbors
from time import time
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor

modelParms = dict(
  nnr=dict(
    n_neighbors=60,
    weights='uniform',  # 'uniform','distance'
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None,
),
  gpr=dict(
    kernel=None,
    alpha=5.0,
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=5,
    normalize_y=True,
    copy_X_train=True,
    random_state=None,
  ),
  svr = dict(
    C=5.0,
    cache_size=500,
    coef0=0.0,
    degree=3,
    epsilon=0.1,
    gamma=2.0,
    kernel='rbf',
    max_iter=-1,
    shrinking=True,
    tol=0.001,
    verbose=False,
  ),
  mlp = dict(
    hidden_layer_sizes=(32,),
    activation="tanh",
    solver='adam',
    alpha=1e-06,
    batch_size='auto',
    learning_rate="invscaling", # ""constant",
    learning_rate_init=0.01,
    power_t=0.1,  # 0.5,
    max_iter=500,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=True,
    validation_fraction=0.2,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
  ),
    rfr = dict(
        n_estimators=30,
        max_features= 'sqrt',  # 'log2'  'sqrt',
        max_depth=10,
        oob_score=True
    )
)
pca_components = 0 # 14
whiten = False
modelType = "nnr"
validation_fraction=0.2

def get_parm_name( svmArgs: Dict ) -> str:
    kernel = svmArgs["kernel"]
    if kernel.lower() == "rbf": return "gamma"
    else: return "coef0"

def getModel( modelType, p0, p1 ):
    params = {}
    if modelType == "svr":
        params['C'] = p0
        params['gamma'] = p1
        print(f"Fitting {modelType} Model, kernel={params['kernel']}, C={p0}, gamma={p1}")
    elif modelType == "mlp":
        params['alpha'] = p0
        params['power_t'] = p1
        print(f"Fitting {modelType} Model, alpha={p0}")
    elif modelType == "gpr":
        params['alpha'] = p0
        print(f"Fitting {modelType} Model, alpha={p0}")
    elif modelType == "nnr":
        params['n_neighbors'] = p0
        print(f"Fitting {modelType} Model, n_neighbors={p0}")
    elif modelType == "rfr":
        print(f"Fitting {modelType} Model")
    else: raise Exception( f" Unknown Model type: {modelType}")
    return getParameterizedModel(  modelType, **params )

def getParameterizedModel( modelType, **newParams ):
    defaultparams = modelParms[modelType]
    params = { **defaultparams } if not newParams else { **defaultparams, **newParams }
    if modelType == "svr":
        return svm.SVR(**params)
    elif modelType == "mlp":
        return MLPRegressor(**params)
    elif modelType == "gpr":
        return GaussianProcessRegressor(**params)
    elif modelType == "nnr":
        return neighbors.KNeighborsRegressor (**params)
    elif modelType == "rfr":
        return RandomForestRegressor(**params)
    else: raise Exception( f" Unknown Model type: {modelType}")

if __name__ == '__main__':
    print("Reading Data")
    x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    x_valid: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )
    mParams = modelParms[modelType]

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

    model_handles_validation =  mParams.pop('model_handles_validation')
    if model_handles_validation:
        xfit, yfit = x_data_norm, y_data
    else:
        xfit, yfit = x_train, y_train

    ref_mse = math.sqrt( (y_test*y_test).mean() )
    print( f" y_train MSE = {ref_mse} MAX={y_test.max()}")

    ens_min_loss = sys.float_info.max
    best_model = None
    best_prediction_validation = None

    p0_range = mParams.pop('p0_range')
    p1_range = mParams.pop('p1_range')
    results = {}
    best_p0 = None
    best_p1 = None

    for p0 in p0_range:
        for p1 in p1_range:
            model = getModel( modelType, p0, p1 )
            model.fit( xfit, yfit )
            prediction_validation = model.predict( x_test )
            diff = prediction_validation - y_test
            validation_loss = math.sqrt( (diff*diff).mean() )
            print(f" --> loss={validation_loss}")

            if validation_loss < ens_min_loss:
                ens_min_loss = validation_loss
                best_model = model
                best_prediction_validation = prediction_validation
                best_p0 = p0
                best_p1 = p1

    best_prediction_training = best_model.predict( x_train )
    model_label = "-".join([ modelType, str(best_p0), str(best_p1)] )
    print( f"Plotting results: loss={ens_min_loss}, model={model_label}")
    fig = plt.figure()
    # fig.suptitle( "Performance Plots: Target (blue) vs Prediction (red)", fontsize=12 )

    diff = y_train - best_prediction_training
    mse = math.sqrt((diff * diff).mean())
    ax0 = plt.subplot("211")
    ax0.set_title( f"{model_label} Training Data MSE = {mse:.2f} ")
    xaxis = range(best_prediction_training.shape[0])
    ax0.plot(xaxis, y_train, "b--", label="validation data")
    ax0.plot(xaxis, best_prediction_training, "r--", label="prediction")
    ax0.legend()

    diff = y_test - best_prediction_validation
    ref_mse = math.sqrt( (y_test*y_test).mean() )
    mse = math.sqrt((diff * diff).mean())
    print( f" REF MSE = {ref_mse} ")
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