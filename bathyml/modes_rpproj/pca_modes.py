import os, math, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from geoproc.plot.bar import MultiBar

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
whiten = False

# band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIRB']
# RatioPairList = [(b, t) for b in range(2, 8) for t in range(3, 8) if b < t]
# band_names = band_names + ['Ratio' + str(RatioPairList[i][0]) + '_' + str(RatioPairList[i][1]) for i in range(len(RatioPairList))]

band_names = [ 'b1_LC8_075', 'b2_LC8_075', 'b3_LC8_075', 'b4_LC8_075', 'b5_LC8_075', 'b6_LC8_075', 'b7_LC8_075', 'b8_LC8_075', 'b9_LC8_075', 'b10_LC8_07',
               'b11_LC8_07', 'b12_LC8_07', 'b13_LC8_07', 'b14_LC8_07', 'b15_LC8_07', 'b16_LC8_07', 'b17_LC8_07', 'b18_LC8_07', 'b19_LC8_07', 'b20_LC8_07',
               'b21_LC8_07', 'b22_LC8_07', 'b23_LC8_07', 'b24_LC8_07', 'b25_LC8_07', 'b26_LC8_07', 'b27_LC8_07', 'b28_LC8_07', 'b29_LC8_07', 'b30_LC8_07',
               'b31_LC8_07', 'b32_LC8_07', 'b33_LC8_07', 'b34_LC8_07', 'b35_LC8_07' ]

datafile = os.path.join(ddir, 'pts_merged_final.csv' )
usecols = [1] + list(range(3,len(band_names)+3))
dataArray: np.ndarray = np.loadtxt( datafile, delimiter=",", skiprows=1, usecols= usecols )
zd: np.ndarray = dataArray[:,0]
colorData = dataArray[:,1:]
cnorm =  preprocessing.scale( colorData )

pca = PCA(n_components=3, whiten=whiten)
color_point_data_pca = pca.fit(cnorm).transform(cnorm)
cx = color_point_data_pca[:, 0]
cy = color_point_data_pca[:, 1]

barplots = MultiBar("PCA Modes", band_names )

for iC in range(pca.components_.shape[0]):
    ev = pca.explained_variance_ratio_[iC]
    barplots.addPlot(f"PCA Mode {iC}, EV={ev*100:.1f}%" , pca.components_[iC] )

barplots.show()