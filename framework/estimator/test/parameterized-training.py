from bathyml.common.data import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from framework.estimator.base import EstimatorBase
import pandas as pd

scratchDir = os.environ.get( "ILSCRATCH", os.path.expanduser("~/ILAB/scratch") )
outDir = os.path.join( scratchDir, "results", "WaterMapping" )
if not os.path.exists(outDir): os.makedirs( outDir )
version= 0
verbose = False
make_plots = False
nFolds= 5
modelTypes = [ "mlp", "rf", "svr", "nnr" ]

parameters = dict(
    mlp=dict( max_iter=250, learning_rate="constant", solver="adam", early_stopping=False ),
    rf=dict(n_estimators=70, max_depth=20),
    svr=dict(C=5.0, gamma=0.5),
    nnr=dict( n_neighbors=5, weights='distance' ),
)

def mean_abs_error( x: np.ndarray, y: np.ndarray ):
    return np.mean( np.abs( x-y ), axis=0 )

if __name__ == '__main__':
    print("Reading Data")
    pts_data, x_data_raw, y_data = read_csv_data( "pts_merged_final.csv"  )
    x_data_norm = preprocessing.scale( x_data_raw )
    mseCols = ['mse_train', 'mse_trainC', 'mse_test', 'mse_testC']
    scoreCols = [ 'trainScore', 'testScore', 'ConstantModel' ]
    outputCols = [ 'Algorithm', "TestID", "OID", "FID", "Subset", "Actual", "Prediction" ]
    global_score_table = IterativeTable( cols=scoreCols )
    results_table = IterativeTable( cols=outputCols )

    for modelType in modelTypes:
        score_table = IterativeTable( cols=mseCols )
        for validFold in range( nFolds-1 ):
            validation_fraction = 1.0 / nFolds if nFolds > 1 else None
            if make_plots:
                fig, ax = plt.subplots(2,1)
            estimator: EstimatorBase = EstimatorBase.new( modelType )
            print( f"Executing {modelType} estimator, validation_fraction={validation_fraction}, fold = {validFold}, parameterList: {estimator.parameterList}" )
            pts_train, pts_test, x_train, x_test, y_train, y_test = getKFoldSplit(pts_data, x_data_norm, y_data, nFolds, validFold)
            estimator.update_parameters( validFold=validFold, validation_fraction=validation_fraction, **parameters[modelType] )
            x_train_valid, y_train_valid = (np.concatenate( [x_train, x_test] ), np.concatenate( [y_train, y_test] ) ) if ( modelType == "mlp" and parameters['early_stopping'] )  else (x_train, y_train)
            estimator.fit( x_train_valid, y_train_valid )
            model_mean, model_std  =  y_train.mean(), y_train.std()
            const_model_train = np.full( y_train.shape, model_mean )
            const_model_test = np.full(y_test.shape, model_mean )
            random_model_train = np.random.normal( model_mean, model_std, y_train.shape )
            random_model_test = np.random.normal( model_mean, model_std, y_test.shape )

            test_prediction = estimator.predict(x_test)
            train_prediction = estimator.predict(x_train)

            mse_train =  mean_abs_error( y_train, train_prediction )
            mse_trainC = mean_abs_error( y_train, const_model_train )
            mse_trainR = mean_abs_error( y_train, random_model_train )

            if make_plots:
                ax[0].set_title( f"{modelType} Train Data MSE = {mse_train:.2f}: C={mse_trainC:.2f} R={mse_trainR:.2f} ")
                xaxis = range(train_prediction.shape[0])
                ax[0].plot(xaxis, y_train, "b--", label="train data")
                ax[0].plot(xaxis, train_prediction, "r--", label="prediction")
                ax[0].legend()

            mse_test =  mean_abs_error( y_test, test_prediction )
            mse_testC = mean_abs_error( y_test, const_model_test )
            mse_testR = mean_abs_error( y_test, random_model_test )

            if make_plots:
                ax[1].set_title( f"{modelType} Test Data MSE = {mse_test:.2f}: C={mse_testC:.2f} R={mse_testR:.2f} ")
                xaxis = range(test_prediction.shape[0])
                ax[1].plot(xaxis, y_test, "b--", label="test data")
                ax[1].plot(xaxis, test_prediction, "r--", label="prediction")
                ax[1].legend()

            for idx in range(pts_train.shape[0]):
                pts = pts_train[idx]
                results_table.add_row(data = [modelType, validFold, pts[0], pts[1], "train", f"{y_train[idx]:.3f}", f"{train_prediction[idx]:.3f}" ] )

            for idx in range(pts_test.shape[0]):
                pts = pts_test[idx]
                results_table.add_row(data = [modelType, validFold, pts[0], pts[1], "test", y_test[idx], test_prediction[idx]])

            if make_plots:
                plt.tight_layout()
                outFile =  os.path.join( outDir, f"plots{version}-{modelType}-{validFold}.png" )
                print(f"Saving plots to {outFile} ")
                plt.savefig( outFile )
                plt.close( fig )

            print( f"Performance {modelType}-{validFold}: ")
            print( f" ----> TRAIN SCORE: {mse_trainC/mse_train:.2f} [ MSE= {mse_train:.2f}: C={mse_trainC:.2f} R={mse_trainR:.2f} ]")
            print( f" ----> TEST SCORE:  {mse_testC/mse_test:.2f} [ MSE= {mse_test:.2f}: C={mse_testC:.2f} R={mse_testR:.2f} ]")
            score_table.add_row( validFold, [mse_train, mse_trainC, mse_test, mse_testC] )

        print(f" AVE performance[{modelType}]: {parameters[modelType]} " )
        sums: pd.DataFrame = score_table.get_sums()
        scores = [ min( sums['mse_trainC']/sums['mse_train'], 3.0 ), sums['mse_testC']/sums['mse_test'], 1.0 ]
        print( f" SCORES: train= {scores[0]}, test= {scores[1]} " )
        if make_plots: global_score_table.add_row( modelType.upper(), scores )

    results_path = os.path.join( outDir, f"results-{version}.csv" )
    results_table.to_csv( results_path, index=False )
    print( f"Saved results to '{results_path}'")

    if make_plots:
        scores_table = global_score_table.get_table()
        scores_table.to_csv( os.path.join( outDir, f"scores-{version}.csv" ) )
        scores_table.plot.bar()
        plt.show()




