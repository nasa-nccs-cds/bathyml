import os, math, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Optional, Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn import preprocessing
from bathyml.common.training import getParameterizedModel
from mpl_toolkits.mplot3d import Axes3D
from functools import total_ordering

@total_ordering
class Cluster:

    def __init__(self, index: int, color_centroid: np.ndarray, color_data: np.ndarray, loc_data: np.ndarray, height_data: np.ndarray ):
        self.centroid: np.ndarray = color_centroid
        self.color_data: np.ndarray = color_data
        self.loc_data: np.ndarray = loc_data
        self.height_data = height_data
        self.index: int = index
        self.rank: float = 0.0

    def __lt__(self, other: "Cluster" )-> bool:
        return self.rank < other.rank

    def __eq__(self, other: "Cluster" )-> bool:
        return self.rank == other.rank

    def eval( self, centroid: np.ndarray ):
        diff: np.ndarray = self.centroid - centroid
        self.rank = np.sum( diff*diff, axis=0 )

def read_csv_data( fileName: str, nBands: int = 0 ) -> np.ndarray:
    file_path: str = os.path.join( ddir, "csv", fileName )
    raw_data_array: np.ndarray = np.loadtxt( file_path, delimiter=',')
    if (nBands > 0): raw_data_array = raw_data_array[:,:nBands]
    return raw_data_array

def compute_clusters(cluster_indices: np.ndarray, data: np.ndarray) -> List[Cluster]:
    clusterDict = {}
    for iP in range( data.shape[0]):
        iC = cluster_indices[iP]
        colors, locs, heights = clusterDict.setdefault( iC, ([],[],[]) )
        colors.append( dataArray[ iP, 3:nBands + 3] )
        heights.append( dataArray[iP,2] )
        locs.append( dataArray[iP, 0:2] )
    clusters: List[Cluster] = []
    for cindex, (colors, locs, heights ) in clusterDict.items():
        csum = sum( colors )
        cmean = csum/len( colors )
        clusters.append( Cluster( cindex, cmean, np.vstack(colors), np.vstack(locs),  np.vstack(heights) ) )
    return clusters

def sort_clusters( clusters: List[Cluster], centroid: np.ndarray ) -> List[Cluster]:
    for cluster in clusters: cluster.eval( centroid )
    clusters.sort()
    return clusters

def get_training_clusters( clusters: List[Cluster], centroid: np.ndarray, n_train_clusters: int) -> List[Cluster]:
    sorted_clusters = sort_clusters( clusters, centroid )
    return sorted_clusters[1:n_train_clusters+1]

def get_training_set( clusters: List[Cluster] ) -> Tuple[np.ndarray,np.ndarray]:
    xdata, ydata = [], []
    for cluster in clusters:
        xdata.append( cluster.color_data)
        ydata.append( cluster.height_data )
    return np.concatenate(xdata, axis=0), np.concatenate(ydata, axis=0)


thisDir = os.path.dirname(os.path.abspath(__file__))
ddir = os.path.join(os.path.dirname(os.path.dirname(thisDir)), "data", "csv")
nBands = 21
whiten = False
typeLabel = "train"
modelType = "rfr"
target_cluster_index = 5
show_cluster_locs = False
model_label = "-".join( [modelType, str(target_cluster_index)] )

datafile = os.path.join(ddir, f'lake_data_{typeLabel}.csv' )
dataArray: np.ndarray = np.loadtxt( datafile, delimiter=",")
xyData =  dataArray[:,0:2]

db = DBSCAN(eps=600.0, min_samples=8).fit(xyData)
cluster_indices: np.ndarray = db.labels_
cmaxval = cluster_indices.max()
ccolors = (db.labels_ + 1.0) * (255.0 / cmaxval)

loc0: np.ndarray = dataArray[:,0]
loc1: np.ndarray = dataArray[:,1]
zd: np.ndarray = dataArray[:,2]

if show_cluster_locs:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.scatter(loc0, loc1, c=ccolors, cmap="jet", s=3 )
    plt.show()

colorData = dataArray[:,3:nBands+3]
cnorm =  preprocessing.scale( colorData )
color_clusters: List[Cluster] = compute_clusters( cluster_indices, dataArray )

targetCluster = color_clusters[target_cluster_index]

sorted_clusters = get_training_clusters( color_clusters, targetCluster.centroid, 10 )

x_train, y_train = get_training_set( sorted_clusters )
x_test, y_test   = get_training_set( [targetCluster] )

model = getParameterizedModel( modelType )
model.fit( x_train, y_train.ravel() )
prediction_training = model.predict(x_train)
prediction_validation = model.predict(x_test)

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





