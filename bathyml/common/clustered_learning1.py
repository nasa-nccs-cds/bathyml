import os, math, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bathyml.common.training import getParameterizedModel
from mpl_toolkits.mplot3d import Axes3D

def read_csv_data( fileName: str, nBands: int = 0 ) -> np.ndarray:
    file_path: str = os.path.join( ddir, "csv", fileName )
    raw_data_array: np.ndarray = np.loadtxt( file_path, delimiter=',')
    if (nBands > 0): raw_data_array = raw_data_array[:,:nBands]
    return raw_data_array

def compute_cluster_centroids(clusters: np.ndarray, data: np.ndarray) -> np.ndarray:
    clusterDict = {}
    for iP in range( data.shape[0]):
        iC = clusters[iP]
        cluster = clusterDict.setdefault( iC, [] )
        cluster.append( data[iP] )
    centroids = []
    for cluster in clusterDict.values():
        csum = sum( cluster )
        cmean = csum/len( cluster )
        centroids.append( cmean )
    return np.stack( centroids, axis=0 )

thisDir = os.path.dirname(os.path.abspath(__file__))
ddir = os.path.join(os.path.dirname(os.path.dirname(thisDir)), "data", "csv")
nBands = 21
whiten = False
typeLabel = "train"
validation_fraction = 0.2
clusterIndex = 1
modelType = "mlp"

datafile = os.path.join(ddir, f'lake_data_{typeLabel}.csv' )
dataArray: np.ndarray = np.loadtxt( datafile, delimiter=",")
xyData =  dataArray[:,0:2]

db = DBSCAN(eps=600.0, min_samples=8).fit(xyData)
clusters: np.ndarray = db.labels_
cmaxval = clusters.max()
ccolors = (db.labels_ + 1.0) * (255.0 / cmaxval)

loc0: np.ndarray = dataArray[:,0]
loc1: np.ndarray = dataArray[:,1]
zd: np.ndarray = dataArray[:,2]

colorData = dataArray[:,3:nBands+3]
cnorm =  preprocessing.scale( colorData )

pca = PCA(n_components=3, whiten=whiten)
color_point_data_pca = pca.fit(colorData).transform(colorData)
cx = color_point_data_pca[:, 0]
cy = color_point_data_pca[:, 1]

cdx, cdy = ( cx + 674.253 )/300.0, ( cy + 321.075 )/300.0
mask0 = (cdx*cdx + cdy*cdy) < 1.0
mask = mask0 if clusterIndex == 0 else mask0 == False

xdata = cnorm[ mask ]
ydata = zd[ mask ]
NValidationElems = int(round(xdata.shape[0] * validation_fraction))
NTrainingElems = xdata.shape[0] - NValidationElems
model_label = "-".join([modelType, str(clusterIndex), str(validation_fraction)])

x_train = xdata[:NTrainingElems]
x_test = xdata[NTrainingElems:]
y_train = ydata[:NTrainingElems]
y_test = ydata[NTrainingElems:]

model = getParameterizedModel( modelType )
model.fit(x_train, y_train)
prediction_training = model.predict(x_train)
prediction_validation = model.predict(x_test)
diff = prediction_validation - y_test
validation_loss = math.sqrt((diff * diff).mean())
print(f" --> loss={validation_loss}")

diff = y_train - prediction_training
mse = math.sqrt((diff * diff).mean())
ax0 = plt.subplot("211")
ax0.set_title(f"{model_label} Training Data MSE = {mse:.2f} ")
xaxis = range(prediction_training.shape[0])
ax0.plot(xaxis, y_train, "b--", label="validation data")
ax0.plot(xaxis, prediction_training, "r--", label="prediction")
ax0.legend()
plt.show()

diff = y_test - prediction_validation
ref_mse = math.sqrt((y_test * y_test).mean())
mse = math.sqrt((diff * diff).mean())
print(f" REF MSE = {ref_mse} ")
ax1 = plt.subplot("212")
ax1.set_title(f"{model_label} Validation Data MSE = {mse:.2f} ")
xaxis = range(prediction_validation.shape[0])
ax1.plot(xaxis, y_test, "b--", label="training data")
ax1.plot(xaxis,prediction_validation, "r--", label="prediction")
ax1.legend()
plt.show()
