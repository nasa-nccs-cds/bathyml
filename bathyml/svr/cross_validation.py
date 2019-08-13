from bathyml.common.data import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
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

if __name__ == '__main__':
    print("Reading Data")
    x_train: np.ndarray = read_csv_data( "temp_X_train_inter.csv", nBands )
    y_train: np.ndarray = read_csv_data( "temp_Y_train_inter.csv" )
    x_valid: np.ndarray = read_csv_data( "temp_X_test_inter.csv", nBands )
    y_valid: np.ndarray = read_csv_data( "temp_Y_test_inter.csv" )
    x_train_valid, y_train_valid = getTrainingData( x_train, y_train, x_valid, y_valid )

    best_score = -sys.float_info.max
    bestParms = {}
    results = []

    C_range = np.logspace(-4, -2, 3)
    gamma_range = np.logspace(-4, -2, 3)

    for c in C_range:
        for g in gamma_range:
            svmArgs['C'] = c
            svmArgs['gamma'] = g
            model = svm.SVR( **svmArgs )
            print( f"Fitting Model, c={c}, gamma={g}" )
            scores = cross_val_score( model, x_train_valid, y_train_valid, cv=KFold(n_splits=5), scoring="neg_mean_absolute_error" )
            train_score = scores.mean()
            print(f"Accuracy [C={c:0.2f}, G={g:0.2f}]: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

            if train_score > best_score:
                best_score = train_score
                best_model = model
                results.append( train_score )
                bestParms['C'] = c
                bestParms['gamma'] = g

    print(f"BEST CV Accuracy [C={bestParms['C']:0.2f}, G={bestParms['gamma']:0.2f}]: %0.2f" % (best_score) )

    npscores = np.ndarray(results).reshape( C_range.shape[0], gamma_range.shape[0] )
    plt.figure()
    plt.imshow( npscores, interpolation='nearest', cmap="jet" )
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(gamma_range.shape[0]), gamma_range, rotation=45)
    plt.yticks(np.arange(C_range.shape[0]), C_range)
    plt.title('Validation accuracy')
    plt.tight_layout()
    plt.show()