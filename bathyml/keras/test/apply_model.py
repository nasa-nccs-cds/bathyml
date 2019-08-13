import os, gdal, pickle, numpy as np
from bathyml.random_forest.train_apply_RandomForests__shpPostExtrByVal_updated081718 import stack_to_obj
from bathyml.keras.test.neural_network_test_inter import get_model as get_ann_model
gdal.UseExceptions()
gdal.AllRegister()
gdal.UseExceptions()
drvtif = gdal.GetDriverByName("GTiff")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname(HERE) ) ), "data" )
outDir = os.path.join(DATA, "results")
VHRdir = os.path.join( DATA, "image" )


def array_to_tif( inarr, outfile, imgProperties ):
    ( gt, proj, ncols, nrows, ndval0 ) = imgProperties
    drv = drvtif.Create(outfile, ncols, nrows, 1, gdal.GDT_Float32 )
    drv.SetGeoTransform(gt)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).WriteArray(inarr)
    return outfile

def normalize( array: np.ndarray, scalef = 1.5 ):
    ave = array.mean( axis=0 )
    std = array.std( axis=0 )
    scale = scalef * std
    return (array-ave)/scale, scale, std

def apply_model( image_name, modelName):
    VHRstack = os.path.join(VHRdir, image_name)
    (img, imgProperties) = stack_to_obj(VHRstack)
    (gt, proj, ncols, nrows, ndval) = imgProperties  # ndval is nodata val of image stack not sample points

    # Classification of img array and save as image (5 refers to the number of bands in the stack)
    # reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array: np.ndarray = img[:, :, :img.shape[2]].reshape(new_shape)  # 5 is number of layers
    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))
    dmax, dmin = img_as_array.max(), img_as_array.min()
    img_as_array, scale, offset = normalize(img_as_array)

    ndval0 = img_as_array[0,0]
    input_ndval = 0
    out_ndval = 0.0
    img_as_array[ img_as_array[:, :] == ndval0 ] = input_ndval

    # Now predict for each pixel
    model_file = open( os.path.join( outDir, modelName ), "rb")
    model_load = get_ann_model( 0, pickle.load( model_file ) )

    class_prediction: np.ndarray = model_load.predict(img_as_array)
    class_prediction: np.ndarray = class_prediction.reshape(img[:, :, 0].shape).astype(np.float32)
    dmax1, dmin1 = class_prediction.max(), class_prediction.min()
    print(f" pmax1 = {dmax1}, pmin1 = {dmin1}, processed prediciton sample = {class_prediction[0,:10].tolist()}")

    classification = os.path.join( outDir, "{}_{}__classified.tif".format(im_name[:-4], modelName ))
    array_to_tif( class_prediction, classification, imgProperties )
    print("\nWrote map output to {}".format(classification))

    pyplot_array(class_prediction, "Bathymetry prediction")

def pyplot_array( img_array: np.ndarray, title: str ):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title( title )
    cmax, cmin = img_array.max(), img_array.min()
    print( f"pyplot_array: cmax = {cmax}, cmin = {cmin}" )
    im = ax.imshow(img_array, cmap='jet', vmin=cmin, vmax=cmax)
    plt.show()

if __name__ == '__main__':
    im_name = 'LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRatios_6bands.tif'
    model_name = "model_2--64-seg-0.20__08-12-15.42.22"

    apply_model( im_name, model_name )