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

def cluster_data( clusters: np.ndarray, data: np.ndarray ) -> np.ndarray:
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
ddir = os.path.join(os.path.dirname(os.path.dirname(thisDir)), "data")
nBands = 21
pca_components = 3
whiten = False
showSpaceClusters = True
showPCAClusters = True

typeLabel = "train"
outfile = os.path.join(ddir, f'PointXYData-{typeLabel}.csv' )
dataArray: np.ndarray = np.loadtxt( outfile, delimiter=",")
db = DBSCAN(eps=600.0, min_samples=8).fit(dataArray)
clusters: np.ndarray = db.labels_
cmaxval = clusters.max()
ccolors = (db.labels_ + 1.0) * (255.0 / cmaxval)

if showSpaceClusters:
    x = dataArray[:,0]
    y = dataArray[:,1]

    xy_centroids = cluster_data( ccolors, dataArray )
    cx = xy_centroids[:,0]
    cy = xy_centroids[:,1]

    fig, ax = plt.subplots()
    ax.set_title("Data Points")
    ax.scatter( x, y, c=ccolors, cmap="prism"  )
    ax.scatter( cx, cy, c="black", s=1 )

    plt.show()

if showPCAClusters:
    x_train: np.ndarray = read_csv_data("temp_X_train_inter.csv", nBands)
    print( x_train.shape )
    print(x_train[0])
    print(x_train[-1])

    pca = PCA(n_components=pca_components, whiten=whiten)
    color_point_data_pca = pca.fit(x_train).transform(x_train)
    print(f'PCA: explained variance ratio ({pca_components} components): {pca.explained_variance_ratio_}')

    x = color_point_data_pca[:, 0]
    y = color_point_data_pca[:, 1]
    fig, ax = plt.subplots()
    ax.set_title("PCA-Reduced Color Points")
    im = ax.scatter(x, y, s=2)
    plt.show()

    pca = PCA(n_components=pca_components, whiten=whiten)
    centroids = cluster_data(ccolors, x_train)

    color_data_pca = pca.fit(centroids).transform(centroids)
    print(f'PCA: explained variance ratio ({pca_components} components): {pca.explained_variance_ratio_}')

    x = color_data_pca[:, 0]
    y = color_data_pca[:, 1]

    if pca_components == 2:
        fig, ax = plt.subplots()
        ax.set_title("PCA-Reduced Color Centroid Points")
        im = ax.scatter( x, y, s=2  )
    else:
        z = color_data_pca[:, 2]
        kmc = KMeans( n_clusters=4 ).fit( color_data_pca )
        clusters: np.ndarray = kmc.labels_
        maxval = clusters.max()
        colors = ( kmc.labels_ + 1.0 ) * (255.0/maxval)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(x, y, z, s=5, c=colors, cmap="Dark2" )

    plt.show()