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

def cluster( xyData ):
    db = DBSCAN(eps=600.0, min_samples=8).fit(xyData)
    clusters: np.ndarray = db.labels_
    cmaxval = clusters.max()
    ccolors = (db.labels_ + 1.0) * (255.0 / cmaxval)
    return ccolors

thisDir = os.path.dirname(os.path.abspath(__file__))
ddir = os.path.join(os.path.dirname(os.path.dirname(thisDir)), "data", "csv")
nBands = 21
whiten = False
typeLabel = "train"
modelType = "mlp"

band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIRB']
RatioPairList = [(b, t) for b in range(2, 8) for t in range(3, 8) if b < t]
band_names = band_names + ['Ratio' + str(RatioPairList[i][0]) + '_' + str(RatioPairList[i][1]) for i in range(len(RatioPairList))]

datafile = os.path.join(ddir, f'lake_data_{typeLabel}.csv' )
dataArray: np.ndarray = np.loadtxt( datafile, delimiter=",")
xyData =  dataArray[:,0:2]

loc0: np.ndarray = dataArray[:,0]
loc1: np.ndarray = dataArray[:,1]
zd: np.ndarray = dataArray[:,2]

colorData = dataArray[:,3:nBands+3]
cnorm =  preprocessing.scale( colorData )

pca = PCA(n_components=3, whiten=whiten)
color_point_data_pca = pca.fit(cnorm).transform(cnorm)
cx = color_point_data_pca[:, 0]
cy = color_point_data_pca[:, 1]

fig = plt.figure()
fig.suptitle( "PCA Modes", fontsize=12 )
y_pos = np.arange(len(band_names))

for iC in range(pca.components_.shape[0]):
    component = pca.components_[iC]
    ev = pca.explained_variance_ratio_[iC]
    ax = plt.subplot( f"13{iC+1}")
    ax.set_title( f"PCA Mode {iC}, EV={ev*100:.1f}%" )
    ax.barh(y_pos, component, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels( band_names )
    ax.invert_yaxis()

plt.tight_layout()
plt.show()
