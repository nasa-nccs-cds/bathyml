import os, numpy as np
import matplotlib.pyplot as plt

outDir = os.path.expanduser("~/results")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
ddir = os.path.join(DATA, "csv")

def plot_training(subplot):
    rf_prediction_data = os.path.join(ddir, "training_prediction.csv")
    y_training_data =    os.path.join(ddir, "temp_Y_train.csv")

    rf_prediction = np.loadtxt( rf_prediction_data, delimiter=',')
    y_training    = np.loadtxt( y_training_data,    delimiter=',')

    subplot.plot(range(y_training.shape[0]), y_training, "g--", label="y_training" )
    subplot.plot(range(rf_prediction.shape[0]), rf_prediction, "y--", label="rf_training_prediction" )



def plot_test(subplot):
    rf_prediction_data = os.path.join(ddir, "test_prediction.csv")
    y_training_data = os.path.join(ddir, "temp_Y_test.csv")

    rf_prediction = np.loadtxt(rf_prediction_data, delimiter=',')
    y_training = np.loadtxt(y_training_data, delimiter=',')

    subplot.plot(range(y_training.shape[0]), y_training, "b--", label="y_test")
    subplot.plot(range(rf_prediction.shape[0]), rf_prediction, "r--", label="rf_test_prediction")


subplots = plt.subplots(1, 2)
plot_test(subplots[1][0])
plot_training(subplots[1][1])

plt.show()
