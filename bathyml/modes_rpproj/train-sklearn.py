from bathyml.common.data import *
import matplotlib.pyplot as plt
from framework.estimator.base import EstimatorBase
import pandas as pd

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
nVersions = 8
n_inputs = 35
verbose = False
modelType =  "mlp" #[ "mlp", "rf", "svr", "nnr" ]
make_plots = False

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

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv" )
    x_data_norm = EstimatorBase.normalize( x_data_raw[:,0:n_inputs] )

    if make_plots:
        fig, ax = plt.subplots()
    else: fig, ax = None, None

    for modelType in parameters.keys():
        for iVersion in range(nVersions):
            fit_args = dict(
                mlp = dict( write_init_weights  = os.path.join( outDir,  f"init_{modelType}_weights.T{iVersion}.pkl"),
                            write_final_weights = os.path.join( outDir,  f"final_{modelType}_weights.T{iVersion}.pkl")) )
            modParms = parameters[modelType]
            modParms['random_state'] = iVersion
            estimator: EstimatorBase = EstimatorBase.new( modelType )
            estimator.update_parameters( **modParms )
            print( f"Executing {modelType} estimator, parameters: { estimator.instance_parameters.items() } " )
            estimator.fit( x_data_norm, y_data, **fit_args.get(modelType,{}) )
            model_mean  =  y_data.mean()
            const_model_train = np.full( y_data.shape, model_mean )
            print( f"Performance {modelType}: ")

            train_prediction = estimator.predict(x_data_norm)
            mse_train =  mean_squared_error( y_data, train_prediction )
            mse_trainC = mean_squared_error( y_data, const_model_train )

            print( f" ----> TRAIN SCORE: {mse_trainC/mse_train:.2f} [ MSE= {mse_train:.2f}: C={mse_trainC:.2f}  ]")

            saved_model_path = os.path.join( outDir, f"model.{modelType}.T{iVersion}.pkl")
            filehandler = open(saved_model_path,"wb")
            pickle.dump( estimator, filehandler )
            print( f"Saved {modelType}.{iVersion} Estimator to file {saved_model_path}" )


            if make_plots:
                ax.set_title( f"{modelType} Train Data MSE = {mse_train:.2f}: C={mse_trainC:.2f} ")
                xaxis = range(train_prediction.shape[0])
                ax.plot(xaxis, y_data, "b--", label="train data")
                ax.plot(xaxis, train_prediction, "r--", label="prediction")
                ax.plot(xaxis, const_model_train, "r--", label="const_model")
                ax.legend()
                plt.tight_layout()
                # outFile =  os.path.join( outDir, f"plots{iVersion}-{modelType}.png" )
                # print(f"Saving plots to {outFile} ")
                # plt.savefig( outFile )
                plt.show()
                plt.close( fig )


