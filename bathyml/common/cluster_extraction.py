import os, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
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

datafile = os.path.join(ddir, f'lake_data_{typeLabel}.csv' )
dataArray: np.ndarray = np.loadtxt( datafile, delimiter=",")
xyData =  dataArray[:,0:2]

db = DBSCAN(eps=600.0, min_samples=8).fit(xyData)
clusters: np.ndarray = db.labels_
cmaxval = clusters.max()
ccolors = (db.labels_ + 1.0) * (255.0 / cmaxval)

x: np.ndarray = dataArray[:,0]
y: np.ndarray = dataArray[:,1]
zd: np.ndarray = dataArray[:,2]
zdmax = zd.max()
zdcolor = (zd/zdmax)*255.9

colorData = dataArray[:,3:nBands+3]
color_centroids = compute_cluster_centroids( ccolors, colorData )
depth_centroids = compute_cluster_centroids( ccolors, zd )
zcmax = depth_centroids.max()
zccolor = (depth_centroids/zcmax)*255.9

pca = PCA(n_components=3, whiten=whiten)
color_point_data_pca = pca.fit(colorData).transform(colorData)
cx = color_point_data_pca[:, 0]
cy = color_point_data_pca[:, 1]

dx, dy = ( cx + 674.253 )/300.0, ( cy + 321.075 )/300.0
mask = (dx*dx + dy*dy) < 1.0
color_point_data_pca0 = color_point_data_pca[ mask ]
color_point_data_pca1 = color_point_data_pca[ mask == False ]
zdcolor0 = zdcolor[ mask ]
zdcolor1 = zdcolor[ mask == False ]
print(f" pca0.shape = {color_point_data_pca0.shape}, pca1.shape = {color_point_data_pca1.shape}, zdcolor0.shape = {zdcolor0.shape}, zdcolor1.shape = {zdcolor1.shape}")

cx0 = color_point_data_pca0[:, 0]
cy0 = color_point_data_pca0[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(cx0, cy0, c=zdcolor0, cmap="jet", s=3 )
plt.show()

cx1 = color_point_data_pca1[:, 0]
cy1 = color_point_data_pca1[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.scatter(cx1, cy1, c=zdcolor1, cmap="jet", s=3 )
plt.show()


