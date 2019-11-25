from bathyml.common.data import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from framework.estimator.base import EstimatorBase
import pandas as pd

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "Bathymetry" )
if not os.path.exists(outDir): os.makedirs( outDir )
version= 2
verbose = False
make_plots = False
show_plots = False
modelTypes = [ "mlp", "rf", "svr", "nnr" ]

parameters = dict(
    mlp=dict( max_iter=150, learning_rate="constant", solver="adam", early_stopping=False ),
    rf=dict(n_estimators=70, max_depth=20),
    svr=dict(C=5.0, gamma=0.5),
    nnr=dict( n_neighbors=5, weights='distance' ),
)

def mean_abs_error( x: np.ndarray, y: np.ndarray ):
    return np.mean( np.abs( x-y ), axis=0 )

def mean_squared_error( x: np.ndarray, y: np.ndarray ):
    diff =  x-y
    return np.sqrt( np.mean( diff*diff, axis=0 ) )

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv", nbands = 21  )
    x_data_norm = preprocessing.scale( x_data_raw )
    mseCols = ['mse_train', 'mse_trainC', 'mse_test', 'mse_testC']
    scoreCols = [ 'trainScore', 'testScore', 'ConstantModel' ]
    outputCols = [ 'Algorithm', "OID", "FID", "Subset", "Actual", "Prediction" ]
    global_score_table = IterativeTable( cols=scoreCols )
    results_table = IterativeTable( cols=outputCols )

    for modelType in modelTypes:
        score_table = IterativeTable( cols=mseCols )
        modParms = parameters[modelType]
        if make_plots:
            fig, ax = plt.subplots(1)
        estimator: EstimatorBase = EstimatorBase.new( modelType )
        print( f"Executing {modelType} estimator, parameterList: {estimator.parameterList}" )
        estimator.fit( x_data_norm, y_data )
        model_mean, model_std  =  y_data.mean(), y_data.std()
        const_model_train = np.full( y_data.shape, model_mean )
        random_model_train = np.random.normal( model_mean, model_std, y_data.shape )
        print( f"Performance {modelType}: ")

        train_prediction = estimator.predict(x_data_norm)
        mse_train =  mean_squared_error( y_data, train_prediction )
        mse_trainC = mean_squared_error( y_data, const_model_train )
        mse_trainR = mean_squared_error( y_data, random_model_train )

        for idx in range(pts_data.shape[0]):
            pts = pts_data[idx]
            results_table.add_row(data=[modelType, pts[0], pts[1], "train", f"{y_data[idx]:.3f}", f"{train_prediction[idx]:.3f}"])

        print( f" ----> TRAIN SCORE: {mse_trainC/mse_train:.2f} [ MSE= {mse_train:.2f}: C={mse_trainC:.2f} R={mse_trainR:.2f} ]")

        if make_plots:
            ax.set_title( f"{modelType} Train Data MSE = {mse_train:.2f}: C={mse_trainC:.2f} R={mse_trainR:.2f} ")
            xaxis = range(train_prediction.shape[0])
            ax.plot(xaxis, y_data, "b--", label="train data")
            ax.plot(xaxis, train_prediction, "r--", label="prediction")
            ax.plot(xaxis, const_model_train, "r--", label="const_model")
            ax.legend()
            plt.tight_layout()
            outFile =  os.path.join( outDir, f"plots{version}-{modelType}.png" )
            print(f"Saving plots to {outFile} ")
            plt.savefig( outFile )
            if show_plots: plt.show()
            plt.close( fig )

        saved_model_path = os.path.join( outDir, f"model.{modelType}.{version}.pkl")
        filehandler = open(saved_model_path,"wb")
        pickle.dump( estimator, filehandler )
        print( f"Saved {modelType} Estimator to file {saved_model_path}" )

    results_path = os.path.join( outDir, f"results-{version}.csv" )
    results_table.to_csv( results_path, index=False )
    print( f"Saved results to '{results_path}'")





