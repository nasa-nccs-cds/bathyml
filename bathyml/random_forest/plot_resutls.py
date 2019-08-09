import os, numpy as np
import matplotlib.pyplot as plt
from bathyml.random_forest.train_apply_RandomForests__shpPostExtrByVal_updated081718 import folder_output

dtype = "inter"
outDir = os.path.expanduser("~/results")
HERE = os.path.dirname(os.path.abspath(__file__))
ddir = os.path.join( os.path.dirname( os.path.dirname(HERE) ), "data" )

def plot_training(subplot):
    rf_prediction_data = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'training_prediction.csv')
    y_training_data =    os.path.join(ddir, "csv", f"temp_Y_train_{dtype}.csv")

    rf_prediction = np.loadtxt( rf_prediction_data, delimiter=',')
    y_training    = np.loadtxt( y_training_data,    delimiter=',')
    subplot.set_title("Training Data")
    subplot.plot(range(y_training.shape[0]), y_training, "g--", label="y_training" )
    subplot.plot(range(rf_prediction.shape[0]), rf_prediction, "y--", label="rf_training_prediction" )
    subplot.legend()



def plot_test(subplot):
    rf_prediction_data = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'test_prediction.csv')
    y_training_data = os.path.join(ddir, "csv", f"temp_Y_test_{dtype}.csv")

    rf_prediction = np.loadtxt(rf_prediction_data, delimiter=',')
    y_training = np.loadtxt(y_training_data, delimiter=',')
    subplot.set_title( "Verification Data")
    subplot.plot(range(y_training.shape[0]), y_training, "b--", label="y_test")
    subplot.plot(range(rf_prediction.shape[0]), rf_prediction, "r--", label="rf_test_prediction")
    subplot.legend()


subplots = plt.subplots(2, 1)

plot_training(subplots[1][0])
plot_test(subplots[1][1])

plt.show()
