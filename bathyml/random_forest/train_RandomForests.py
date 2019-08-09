# created 11/29/2017

# purpose: To build a Random Forest model with an input shapefile and validation shapefile
# and then save/apply it to a raster

# input csv must have columns for each of the 6 bands (no coastal)
# as well as all possible band ratio combos
# (and so the raster must have these also)

# Inputs:
# 1. Test and Training shp (w field 'FiltAvgDep' or some other depth field, FID field, AND
#       extracted band values in fields labeled: bx_rasID+refl),
# 2. Raster Stack [will also need to change raster bands accordingly...]
#       may also need to change directory where raster stack is located.
# (thesis data pts. per pixel split into 2 random groups - created in thesis py script)

print("starting script...")

# Import GDAL, NumPy, and matplotlib
import sys, os
from osgeo import gdal, gdal_array, ogr
import numpy as np
import matplotlib.pyplot as plt
##%matplotlib inline # IPython
from sklearn.ensemble import RandomForestRegressor  # Classifier # CHANGED bc I have continuous data
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from timeit import default_timer as timer
from bathyml.logging import BathmlLogger
# from rasterstats import point_query
# import rasterio
from scipy import stats

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

##import gdal
##from osgeo.gdal import *
gdal.UseExceptions()  # enable exceptions to report errors
drvtif = gdal.GetDriverByName("GTiff")

n_trees = 100
max_feat = 'sqrt'  # 'log2'
modelName = '{}_{}'.format(n_trees, max_feat)  # 'try3' # to distinguish between parameters of each model

extentName = ''  # sys.argv[1]#'qA' # model number or name, so we can save to different directories
band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIRB']  # no coastal

# FOR USE WITH BAND RATIOS********************************************************
RatioPairList = [(b, t) for b in range(2, 8) for t in range(3, 8) if b < t]  # 8 is number of bands in Landsat MultiSpec
band_names = band_names + ['Ratio' + str(RatioPairList[i][0]) + '_' + str(RatioPairList[i][1]) for i in range(len(RatioPairList))]
# ********************************************************************************

bands = [i for i in range(1, len(band_names) + 1)]  # , 2, 3] # bands of the image stack:
# ,'Coastal2','Blue2', 'Green2','Red2','NIR2', 'SWIR2', 'SWIRB2','Coastal3','Blue3', 'Green3','Red3', 'NIR3', 'SWIR3', 'SWIRB3']

im_name = 'LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRatios_6bands.tif'
# ^21 band SR stack with 6 pure bands (no coastal) and 15 pure band ratios
# 'LC08_L1TP_078010_20170628_20170714_01_T1_sr_mystack_ratios.tif'
# 'LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRatios.tif'#1000LyrStackRatios.tif'
# im_name = raw_input('Please enter filename (w/ extension) of image to model (e.g. where hue vals extracted from): ')
# '2016_RGBstack_077011_20160805_20170222_TOA_LShapes1_Tif'  #must match inras string from CreateXYInputValidshp..py
# ^ used to populate input and validation X and Y csv filenames in main()

thisDir = os.path.dirname(os.path.abspath(__file__))
ddir = os.path.join(os.path.dirname(os.path.dirname(thisDir)), "data")
# im_dir = 'Imagery/2016LayerStack/AllBands'

# VHRdir = ddir+'//2017Landsat_PSSCloudFreeJun_Sep//'+im_name[0:40]+'//'
VHRdir = os.path.join( ddir, "image" )
# /LC08_L1TP_078010_20170628_20170714_01_T1_sr//'
# Cloud50pct/LC08_L1TP_076011_20170630_20170715_01_T1_sr//'
print('current VHR dir:', VHRdir)
# getraw=raw_input("Would you like to enter an alternate filepath to im_name: (y/n)")
# if getraw == 'y':
# VHRdir = raw_input('Please enter file path to image file to model depths from (e.g. /att/nobackup/cesimpso/Imagery//:')


# Get the raster stack and sample data
VHRstack = os.path.join( VHRdir, im_name )
print("VHR stack (VHR dir + im_name):", VHRstack)

rasID = im_name[10:14]  # '7611' #max 3 chars bc going into shp attribute field as bX_[rasID][refl] e.g. b2_805TOA
if 'sr' in im_name:
    refl = 'sr'
elif 'TOA' in im_name:
    refl = 'TOA'
else:
    print('Neither SR nor TOA found in input raster filename.. assigning refl "NA"...')
    refl = 'NA'

"""
yn = raw_input("Would you like to change n_trees or max_feat (from defaults 100; sqrt)? (y/n)")
if yn == 'y' or yn =='Y' or yn == 'yes':
    n_trees = input('n_trees (must be int, e.g. try 100):')
    max_feat = input('max_feat (enter sqrt or log):')
"""

# VHRdir = os.path.join(ddir,'Imagery', '2017Landsat_PSSCloudFreeJun_Sep', 'Cloud50pct','LC08_L1TP_076011_20170630_20170715_01_T1_sr')
# 'Imagery','2017LayerStack','AllBands', 'LC8_075011_20170927_20171013')
# os.path.join(ddir, 'Imagery','CompositeBandsLayerStack_20160805_20170927_20170628')

#
folder_output = im_name[0:-8] + '_' + str(n_trees) + '_' + max_feat  # dont change this..

# 'RandomForestTests' #extension from ddir where to find raster for classification

out_xls_path = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, '')


# where to output summary/accuracy table


def find_elapsed_time(start, end):  # example time = round(find_elapsed_time(start, end),3) where start and end = timer()
    elapsed_min = (end - start) / 60
    return float(elapsed_min)


"""Function to read data stack into img object"""


def stack_to_obj(VHRstack):
    print("now: stack to object")

    img_ds = gdal.Open(VHRstack, gdal.GA_ReadOnly)  # GDAL dataset

    gt = img_ds.GetGeoTransform()
    proj = img_ds.GetProjection()
    ncols = img_ds.RasterXSize
    nrows = img_ds.RasterYSize
    ndval = img_ds.GetRasterBand(1).GetNoDataValue()  # should be -999 for all layers, unless using scene as input
    print('nd val:', ndval, ', type:', type(ndval))
    # ndval = 0.00#float(ndval1)

    imgProperties = (gt, proj, ncols, nrows, ndval)
    print('imgProperties', imgProperties, '\n')
    """ Read data stack into array """
    img = np.zeros((nrows, ncols, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    print('gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType):',
          gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    print('\n and raw data type is:', img_ds.GetRasterBand(1).DataType)
    for b in range(img.shape[2]):
        # the 3rd index of img.shape gives us the number of bands in the stack
        print('\nb: {}'.format(b))
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()  # .astype(np.float32) # GDAL is 1-based while Python is 0-based
    print('done reading stack into array')
    print('deleting img_ds')
    del img_ds
    return (img, imgProperties)


def GetBandValsAsArray(shp, fields, ndval):
    print('\n\n --------- creating band value array ------- \n\n')
    print(type(ndval))
    b = len(bands)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp, 0)
    lyr = dataSource.GetLayer()
    dt = type(lyr[0].GetField('b2_0760NA'))  # 'b'+str(1)+'_'+rasID+refl))
    print('data type of b2_0760NA:', dt)

    # outar = np.zeros((len(lyr), b+1)).astype(type(dt))
    # print outar.shape,'\n shape of zeros array to fill with BVs, and datatype is: ', outar.dtype, '=data type'
    # print type(src_ds)

    for i in range(len(fields)):
        # outar[:,i]=GetFieldAsArr(shp, fields[i]).ravel()
        if i == 0:
            out0 = GetFieldAsArr(shp, fields[i])  # .ravel()
        else:
            out0 = np.column_stack((out0, GetFieldAsArr(shp, fields[i])))  # .ravel()))
        # print 'shape and dtype of out0',out0.shape, out0.dtype

    out0 = np.column_stack((out0, GetFieldAsArr(shp, 'myFID')))
    # print out0

    # print "Ex get field as array", GetFieldAsArr(shp, 'myFID').ravel()
    # outar[:,b] = GetFieldAsArr(shp, 'myFID').ravel() #fid_list #last column in out array is FID vals

    depth = GetFieldAsArr(shp, 'FiltAvgDep').ravel()  # !!! ENSURE THIS FIELD IS CORRECT

    # print 'current dtype of outar', outar.dtype
    # fl_outar=outar.astype(np.float32)
    # bands_with_fids=fl_outar[:,:b+1]
    # depth= fl_outar[:,b+1]
    print('shape of x array (bands + FID):', out0.shape, '\nshape of y (depth) array:', depth.shape)
    xar, yar = RemoveNoDataFromArs(out0, depth, ndval)
    print('new shape of x and y arrays after no data values removed:', xar.shape, yar.shape)
    print(xar.dtype, 'is dtype of xarray train and tests')
    print(yar.dtype, "=data type of y array of y train and tests", 'prev dtype of yar was', depth.dtype)
    del dataSource
    return xar, yar


def GetFieldAsArr(shp, fieldname):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp, 0)
    layer = dataSource.GetLayer()
    featureCount = layer.GetFeatureCount()
    print('feature count of shp:', featureCount)
    y = []
    # y = np.zeros((featureCount,1))
    # print y.shape,"=y.shape"

    for feature in layer:
        # print fieldname,"feature is:",feature.GetField(fieldname)
        y.append(feature.GetField(fieldname))  # ogr.GetDriverByName('ESRI Shapefile').Open(shp,0).GetLayer()[x].GetField(field_name)
    y = np.array(y)
    layer.ResetReading()
    # print y.dtype,"is data type of shp",fieldname, "e.g.", y[0]

    return y  # .astype(np.float32)


def RemoveNoDataFromArs(ar1, ar2, ndval):
    xarray = ar1.copy()
    yarray = ar2.copy()
    print('removing rows of ndval:', ndval, 'from x and y arrays, which have len:', len(xarray))
    indexlist = []
    length = len(xarray) - 1
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if xarray[row_index][0] == ndval or xarray[row_index][0] == 'nan' or type(xarray[row_index][0]) is None or np.isnan(xarray[row_index][0]) or \
                xarray[row_index][0] == -9999 or xarray[row_index][0] == -8999:
            indexlist.append(row_index)

    print('length of index list:', len(indexlist))

    testlist1 = []
    testlist2 = []
    testlist3 = []
    testlist4 = []
    testlist5 = []
    testlist6 = []
    # 1
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if xarray[row_index][
            0] == ndval:  # or xarray[row_index][0] == 'nan' or type(xarray[row_index][0]) is None or np.isnan(xarray[row_index][0]) or xarray[row_index][0] == -9999 or xarray[row_index][0] == -8999:
            testlist1.append(row_index)
    print(len(testlist1), 'is length of testlist1 with ==ndval')
    # 2
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if xarray[row_index][
            0] == 'nan':  # or type(xarray[row_index][0]) is None or np.isnan(xarray[row_index][0]) or xarray[row_index][0] == -9999 or xarray[row_index][0] == -8999:
            testlist2.append(row_index)
    print(len(testlist2), 'is len of testlist2 with == nan')
    # 3
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if type(xarray[row_index][0]) is None:  # or np.isnan(xarray[row_index][0]) or xarray[row_index][0] == -9999 or xarray[row_index][0] == -8999:
            testlist3.append(row_index)
    print(len(testlist3), 'is len of testlist3 with type is None')
    # 4
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if np.isnan(xarray[row_index][0]):  # or xarray[row_index][0] == -9999 or xarray[row_index][0] == -8999:
            testlist4.append(row_index)
    print(len(testlist4), 'is len of testlist4 with np.isnan')
    # 5
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if xarray[row_index][0] == -9999:  # or xarray[row_index][0] == -8999:
            testlist5.append(row_index)
    print(len(testlist5), 'is len of testlist5 with == -9999')
    # 6
    for row_index in range(length, -1, -1):  # iterate backwards through array
        if xarray[row_index][0] == -8999:
            testlist6.append(row_index)
    print(len(testlist6), 'is len of testlist6 with = -8999')

    for each in indexlist:
        xarray = np.delete(xarray, each, 0)
        yarray = np.delete(yarray, each, 0)

    return xarray, yarray


"""Function to write final classification to tiff"""


def array_to_tif(inarr, outfile, imgProperties):
    print("now running: array_to_tiff")

    # get properties from input
    (gt, proj, ncols, nrows, ndval) = imgProperties
    print(ndval)

    drv = drvtif.Create(outfile, ncols, nrows, 1, 6)  # , options = [ 'COMPRESS=LZW' ]) #6 is gdal.GDT_Float32
    # 1= number of bands (i think) and 3 = Data Type (16 bit signed)
    print('almost done w array to tif..')
    drv.SetGeoTransform(gt)
    drv.SetProjection(proj)
    print('About to set ND value as')
    try:
        try:
            print(ndval)  # '-9999.0?'
            drv.GetRasterBand(1).SetNoDataValue(ndval)

        except:
            print('ndval of -9999.0 throws an exception')
            drv.GetRasterBand(1).SetNoDataValue(-9999.0)

    except:
        print('ndval set to ndval throws an exception')
        drv.GetRasterBand(1).SetNoDataValue(float(0))

    drv.GetRasterBand(1).WriteArray(inarr)

    return outfile


"""Function to run diagnostics on model"""


def run_diagnostics(model_save, X, y,
                    fid_valid):  # where model is the model object, X and y are training sets, fid_valid is the list of validation depths' fids
    print("now running: run_diagnostics")
    # load model for use:
    print("\nLoading model from {} for cross-val".format(model_save))
    model_load = joblib.load(model_save)  # nd load

    print("\n\nDIAGNOSTICS:\n")

    try:
        print("n_trees = {}".format(n_trees))
        print("max_features = {}\n".format(max_feat))
    except Exception as e:
        print("ERROR: {}\n".format(e))

    # check Out of Bag (OOB) prediction score
    print('Our OOB prediction of accuracy is: {}\n'.format(model_load.oob_score_ * 100))
    print("OOB error: {}\n".format(1 - model_load.oob_score_))

    # check the importance of the bands:
    for b, imp in zip(bands, model_load.feature_importances_):
        print('Band {b} ({name}) importance: {imp}'.format(b=b, name=band_names[b - 1], imp=imp))
    print('')

    """
    # see http://scikit-learn.org/stable/modules/cross_validation.html for how to use rf.score etc
    """
    mod_pred = model_load.predict(X).tolist()

    slope, intercept, rval, pval, stderr = stats.linregress(y, mod_pred)
    print('\n\nStats (validation depths vs. model prediction):\nR2:', rval ** 2)
    print('p value:', pval)
    print('standard error:', stderr)
    print('RMSE:', np.sqrt(((mod_pred - y) ** 2).mean()))

    Dif_MeasMod = [y[i] - mod_pred[i] for i in range(len(y))]
    Dif_MeasMod_4m = [y[i] - mod_pred[i] for i in range(len(y)) if y[i] < 4.0]
    Dif_MeasMod_8m = [y[i] - mod_pred[i] for i in range(len(y)) if y[i] > 8.0]

    AbsDif_MeasMod = [abs(i) for i in Dif_MeasMod]
    AbsDif_MeasMod_4m = [abs(i) for i in Dif_MeasMod_4m]
    AbsDif_MeasMod_8m = [abs(i) for i in Dif_MeasMod_8m]

    print('Mean Difference between Meas. and Modeled Depth:', np.mean(Dif_MeasMod))
    print('Mean Difference between Absolute Meas. and Modeled Depth:', np.mean(AbsDif_MeasMod))
    print('Median Difference between Meas. and Modeled Depth:', np.median(Dif_MeasMod))
    print('Median Difference between Absolute Meas. and Modeled Depth:', np.median(AbsDif_MeasMod))

    print('Mean Difference between Meas. and Modeled Depth (Meas Depth < 4 m):', np.mean(Dif_MeasMod_4m))
    print('Mean Difference between Absolute Meas. and Modeled Depth (Meas Depth < 4 m):', np.mean(AbsDif_MeasMod_4m))
    print('Median Difference between Meas. and Modeled Depth (Meas Depth < 4 m):', np.median(Dif_MeasMod_4m))
    print('Median Difference between Absolute Meas. and Modeled Depth (Meas Depth < 4 m):', np.median(AbsDif_MeasMod_4m))

    print('Mean Difference between Meas. and Modeled Depth (Meas Depth > 8 m):', np.mean(Dif_MeasMod_8m))
    print('Mean Difference between Absolute Meas. and Modeled Depth (Meas Depth > 8 m):', np.mean(AbsDif_MeasMod_8m))
    print('Median Difference between Meas. and Modeled Depth (Meas Depth > 8 m):', np.median(Dif_MeasMod_8m))
    print('Median Difference between Absolute Meas. and Modeled Depth (Meas Depth > 8 m):', np.median(AbsDif_MeasMod_8m))

    perc_list = [y[i] for i in range(len(y)) if AbsDif_MeasMod[i] < (0.1 * (y[i]))]  # absolute difference is less than 10% of true depth
    print('Percentage of depths modeled to within 10% of true depth:', (float(len(perc_list)) * 100.0) / len(y), '%')
    print('Mean and median depth modeled to within 10% of true depth:', np.mean(perc_list), np.median(perc_list))

    perc_list_1m = [y[i] for i in range(len(y)) if AbsDif_MeasMod[i] < 1.0]
    print('Percentage of depths modeled to within 1m of true depth:', (len(perc_list_1m) / float(len(y))) * 100., '%')

    try:
        plt.plot(y, mod_pred, 'o', markerfacecolor='None')  # measured on x, modeled on y
        testar = np.linspace(0, 20)
        plt.plot(testar, testar * slope + intercept)
        plt.xlabel("Measured Depth (m)")
        plt.ylabel("RFA Predicted Depth (m)")
        plt.savefig(out_xls_path + 'TruthVPred_graph.png')
    except:
        'ERROR: graph not created or saved..'

    # dont really know if this is applicable for 2 classes but try it anyway:
    # Setup a dataframe -- just like R
    df = pd.DataFrame({'0FID': fid_valid.ravel(), '1Truth': y.ravel().tolist(), '2RFA predicts': mod_pred})  # xtest, ytest
    # **** Need to create a new y with validation points, like we did with y in the function below
    df.to_excel(out_xls_path + 'Table_RFA_' + str(n_trees) + max_feat + '.xls', header=True, index=False)
    # (make roi be valid sites array instead of training)
    # try:

    """print type(model_load.predict(X)), model_load.predict(X).dtype

    df['predict'] = model_load.predict(X).tolist()
    print 'no error?'

    ## #############
    except:
        print 'an error...'
        try:
            df['truth']=y.tolist()
            df['predict']=model_load.predict(X)
        except:
            raise"""

    # Cross-tabulate predictions
    # print pd.crosstab(df['truth'], df['predict'], margins=True)


##    print "Other:"
##    print model.criterion
##    print model.estimator_params
##    print model.score
##    print model.feature_importances_
##    print ''


def apply_model(img, imgProperties, classDir, model_save):  # VHR stack we are applying model to, output dir, and saved model
    # should I use a stack that only covers lakes and all other pixels have nodata value?
    print("now running 'apply_model'")

    (gt, proj, ncols, nrows, ndval) = imgProperties  # ndval is nodata val of image stack not sample points

    # print img
    print(img.shape)
    print(np.unique(img))

    # Classification of img array and save as image (5 refers to the number of bands in the stack)
    # reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape)  # 5 is number of layers
    ##    print img_as_array.shape # (192515625, 5)
    ##    print np.unique(img_as_array) # [ -999  -149  -146 ..., 14425 14530 14563]

    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    print("\nLoading model from {}".format(model_save))
    model_load = joblib.load(model_save)  # nd load
    print("\nModel information:\n{}".format(model_load))

    # Now predict for each pixel
    class_prediction = model_load.predict(img_as_array)

    # * at some point may need to convert values that were -999 in img array back to -999, depending on what rf does to those areas

    ##    print img[:, :, 0].shape # (13875, 13875)
    ##    print img[:, :, 0]

    # Reshape our classification map and convert to float32 (prevoiusly in version: 16-bit signed int)
    class_prediction = class_prediction.reshape(img[:, :, 0].shape).astype(np.float32)  # .astype(np.int16) #CHANGED: I dont want it to be int type

    ##    print class_prediction # numpy array? what?
    ##    print class_prediction.shape # (13875, 13875)
    ##    print class_prediction.dtype #uint8
    ##    print np.unique(class_prediction) # [1 2]
    ##    print img.shape # (13875, 13875, 5)
    ##    print np.unique(img) # [ -999  -149  -146 ..., 14425 14530 14563]

    # Now we need to convert existing NoData values back to NoData (-999, or 0 if at scene-level)
    class_prediction[img[:, :, 0] == ndval] = ndval
    # just chose the 0th index to find where noData values are (should be the same for all MS layers, not ure about PAn)

    # ?? or should I jsut have ndval be 0? or -9999? or Nodata?

    # use from old method to save to tif
    ##    print np.unique(class_prediction)
    ##    print ndval

    # export classificaiton to tif
    classification = os.path.join(classDir, "{}_{}__{}__classified.tif".format(im_name[:-4], modelName, extentName))
    array_to_tif(class_prediction, classification, imgProperties)
    ##    io.imsave(classification, class_prediction)
    print("\nWrote map output to {}".format(classification))

    return  # ?


"""Function for training the model using training data"""


# To train the model, you need: input text file/csv i.e. x_train and y_train, model output location, model parameters
def train_model(X, y, modelDir, n_trees, max_feat):
    print("training model in progress")
    # y = np.array([y])
    print('y to train model: type is', type(y), 'shape is', y.shape)
    n_samples = np.shape(X)[0]  # the first index of the shape of X will tell us how many sample points
    print('\nWe have {} samples'.format(n_samples))

    # labels = np.unique(y) # now it's the unique values in y array from text file
    # print 'The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels)  #CHANGED: this is irrelevant
    # bc im not looking at unique classes, im looking at continuous data
    # Maggie: will be 2: Water (1) and Not Water (2)
    # Mine: will be RGB and actual depth ?

    print('Our X matrix is sized: {sz}'.format(sz=X.shape))  # hopefully will be (?, 5)
    # for me:(?, number of features e.g 3 for rgb?)
    print('Our y array is sized: {sz}'.format(sz=y.shape))

    """ Now train the model """
    print("\nInitializing model...")
    # rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) # can change/add other settings later
    # CHANGED to below!
    rf = RandomForestRegressor(n_estimators=n_trees, max_features=max_feat, oob_score=True)  # can change/add other settings later
    ravel_y = y.ravel()  # to convert to 1 dimensional array
    print("\nTraining model...")
    rf.fit(X, ravel_y)  # fit model to training data

    print('oob score:', rf.oob_score_)

    # Export model:
    try:
        model_save = os.path.join(modelDir, "model_{}_{}.pkl".format(extentName, modelName))
        joblib.dump(rf, model_save)
    except Exception as e:
        print("An Error: {}".format(e))

    return rf, model_save  # Return output model for application and validation


def get_test_training_sets(inputText):  # CS doesnt use this function

    with open(inputText, 'r') as it:
        cnt = 0
        for r in it.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')

            xx = np.array(rline[0:-1])  # xx is the line except the last entry (class)
            yy = np.array(rline[-1])  # yy is the last entry in the line

            if cnt == 1:
                X = [xx]
                y = yy
            else:
                X = np.vstack((X, xx))
                y = np.append(y, yy)

    # Now we have X and y, but this is not split into validation and training. Do that here:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21)
    # where X_train is an array of all the features (qualities, e.g. RGB values etc.) of the input pixels,
    # y_train is a list of all the depths of the input pixels
    # X_test is an array of all the features of the validation pixels (e.g. RGB vals)
    # and y_test is a list of all the depths of all the validation pixels... 

    return (X_train, X_test, y_train, y_test)


# instead of above definition:
# use this if training and testing RF model with pixels from entire lake..
# otherwise would need to run this script for each lake individually and
# would need to have the 4 input csvs to this function be lake-specific
# rather than inclusive of depth point data from entire region
def get_test_training_sets_CS(X_train_file, X_test_file, y_train_file, y_test_file):  # 1st thing called
    with open(X_train_file, 'r') as xvf:
        cnt = 0
        for r in xvf.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')
            xx = np.array(rline)
            if cnt == 1:
                X = [xx]
            else:
                X = np.vstack((X, xx))
        print('X_train shape:', X.shape)
        X_train = X.astype('float32')  # chane actual script to use this!!

    with open(X_test_file, 'r') as xvf:
        cnt = 0
        for r in xvf.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')
            xx = np.array(rline)
            if cnt == 1:
                X = [xx]
            else:
                X = np.vstack((X, xx))
        print('X_test shape:', X.shape)
        X_test = X.astype('float32')  # chane actual script to use this!!

    with open(y_train_file, 'r') as xvf:
        cnt = 0
        for r in xvf.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')
            xx = np.array(rline)
            if cnt == 1:
                X = [xx]
            else:
                X = np.vstack((X, xx))
        print('y_train shape:', X.shape)
        y_train = X.astype('float32')  # chane actual script to use this!!

    with open(y_test_file, 'r') as xvf:
        cnt = 0
        for r in xvf.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')
            xx = np.array(rline)
            if cnt == 1:
                X = [xx]
            else:
                X = np.vstack((X, xx))
        print('y_test shape:', X.shape)
        y_test = X.astype('float32')  # chane actual script to sue this!!
    # print 'X_train shape:',X_train.shape
    # print 'y_train shape:',y_train.shape
    # print '\n',y_train,'\n'
    return (X_train, X_test, y_train, y_test)


"""Function to convert stack and sample imagery into X and y [not yet split into training/validation]"""
# CHANGE:  won't need this function bc my data already in csv format, however need to get gt, proj, etc.
# to write bathymetry to tiff at end use specs of VHR stack instead
# script never uses this function anyway..?
"""
def convert_img_to_Xy_data(VHRstack, sampleTif): 
    # Given a data stack and sample tiff, we can convert these into X and y for Random Forests
    print "now running: convert img to Xy data"
    ##Read in the raster stack and training data to numpy array: 
    img_ds = gdal.Open(VHRstack, gdal.GA_ReadOnly) # GDAL dataset
    roi_ds = gdal.Open(trainTif, gdal.GA_ReadOnly) #should this bc sampleTif??

    ##get geo metadata so we can write the classification back to tiff 
    gt = img_ds.GetGeoTransform()
    proj = img_ds.GetProjection()
    ncols = img_ds.RasterXSize
    nrows = img_ds.RasterYSize
    ndval = img_ds.GetRasterBand(1).GetNoDataValue() # should be -999 for all layers, unless using scene as input

    imgProperties = (gt, proj, ncols, nrows, ndval)

    # Read data stack into array 
    img = np.zeros((nrows, ncols, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]): # the 3rd index of img.shape gives us the number of bands in the stack
        print '\nb: {}'.format(b)
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray() # GDAL is 1-based while Python is 0-based
        print 'mean of band:',np.mean(img_ds.GetRasterBand(b+1).ReadAsArray())

    #Read Training dataset into numpy array 
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.float32) #used to be uint8 CHNAGED!
    roi_nd = roi_ds.GetRasterBand(1).GetNoDataValue() # get no data value of training dataset
    roi[roi==roi_nd] = 0 # Convert the No Data pixels in raster to 0 for the model

    X = img[roi > 0, :]  # Data Stack pixels
    y = roi[roi > 0]     # Training pixels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 21)

    del img_ds, roi_ds
    return (X_train, X_test, y_train, y_test)
"""


def main():
    print("starting main...")
    start = timer()
    # Set up directories: NEED TO CHANGE VHS and train DIRS (script makes down model, class and log dirs)*********** 
    # /RandomForestsTest' #'/att/gpfsfs/briskfs01/ppl/mwooten3/Myanmar/RandomForests/'
    # ********* CHANGE THIS^: my folder path
    ######*************************************

    bandExtr_fieldList = ['b2_0760NA', 'b3_0760NA', 'b4_0760NA', 'b5_0760NA', 'b6_0760NA', 'b7_0760NA', 'b14_0760NA',
                          'b15_0760NA', 'b16_0760NA', 'b17_0760NA', 'b18_0760NA', 'b19_0760NA', 'b20_0760NA', 'b21_0760NA',
                          'b22_0760NA', 'b23_0760NA', 'b24_0760NA', 'b25_0760NA', 'b26_0760NA', 'b27_0760NA', 'b28_0760NA']

    bandExtr_fieldList1 = ['b1_0780SR', 'b2_0780SR', 'b3_0780SR', 'b4_0780SR', 'b5_0780SR', 'b6_0780SR', 'b7_0780SR',
                           'b8_0780SR', 'b9_0780SR', 'b10_0780SR', 'b11_0780SR', 'b12_0780SR', 'b13_0780SR', 'b14_0780SR',
                           'b15_0780SR', 'b16_0780SR', 'b17_0780SR', 'b18_0780SR', 'b19_0780SR', 'b20_0780SR', 'b21_0780SR']
    # ['b'+str(i)+'_'+rasID+refl for i in xrange(1,len(bands)+1)]

    # bandExtr_fieldList = ['b'+str(i) +'_0780SR' for i in range(1,7)]
    print("Field List:", bandExtr_fieldList)
    print(len(bandExtr_fieldList), 'is length of band extract field list')

    # ? will these be created in loop below? 
    modelDir = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output,
                            'Models')  # , '{}_{}'.format(extentName, modelName)) #C #Model output location
    classDir = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'Classified')  # , extentName) #C #Location for output classification
    # i.e. bathymetry raster created by RF model
    logDir = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'Logs')  # C

    for d in [modelDir, classDir, logDir]:
        os.system('mkdir -p {}'.format(d))

    print("done making dirs")

    # Log processing:
    print("done log processing")

    # os.path.join(VHRdir, im_name)#'DataStack__{}.tif'.format(extentName))
    # ^ Stack of VHR data (Multispec and Pan) //only covers [all North Slope, not just lakes w/ in situ data] lakes! 
    # & make other pixels have no data valu

    # take in shapefile and extract raster values at those points, export band vals to csv to use to train model as inputX
    # not a big deal which shps you input just as long as they're derived from correct InputX_FID and ValidX_FID csvs
    train_shp = os.path.join(ddir, 'gis', 'input_JonesPSS_copy.shp')
    # use copy if you want the 2017 bands extracted from 076011_20170630
    # dont use copy if you want 2017 bands extracted from 078010_20170628
    # 'inputRFA_LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRa.shp')# 'inputRFA_'+im_name[0:-8]+'.shp')#'input_RFA.shp')
    test_shp = y_test_shp = os.path.join(ddir, 'gis', 'valid_JonesPSS_copy.shp')
#    test_shp = y_test_shp = os.path.join(ddir, 'gis', 'input_JonesPSS_copy.shp')
    # 'validRFA_LC08_L1TP_076011_20170630_20170715_01_T1_StackBandsAndRa.shp')#'validRFA_'+im_name[0:-8]+'.shp')#'valid_RFA.shp')
    print(test_shp, 'is test shp')

    (img, imgProperties) = stack_to_obj(VHRstack)

    X_train, y_train = GetBandValsAsArray(shp=train_shp, fields=bandExtr_fieldList, ndval=imgProperties[4])
    print("in main, type of X_train is:", X_train.dtype, y_train.dtype, 'is type of ytrain')

    X_test, y_test = GetBandValsAsArray(shp=test_shp, fields=bandExtr_fieldList, ndval=imgProperties[4])

    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'temp_X_train.csv')
    np.savetxt(outfile, X_train, delimiter=",")
    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'temp_Y_test.csv')
    np.savetxt(outfile, y_test, delimiter=",")
    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'temp_Y_train.csv')
    np.savetxt(outfile, y_train, delimiter=",")
    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'temp_X_test.csv')
    np.savetxt(outfile, X_test, delimiter=",")
    # 1st step: get_test_training_sets
    # IF input is csv file, use this method
    """Before calling the model train or apply, read and configure the inputs into test and train """
    # inputText = '/att/gpfsfs/briskfs01/ppl/mwooten3/Myanmar/RandomForests/TrainingData/Myanmar_5class_training.csv'

    ##    # if input is an image stack and training tif, use this: 
    fid_test = X_test[:, len(bands)].astype(int)  # all rows and the last column (8th col, but index pos is 7)
    X_test = X_test[:, :len(bands)]
    X_train = X_train[:, :len(bands)]

    print('shape of X_train:', X_train.shape)
    print( 'N bands:', bands )

    # Train and apply models:
    print("Building model with n_trees={} and max_feat={}...".format(n_trees, max_feat))
    rf, model_save = train_model(X_train, y_train, modelDir, n_trees, max_feat)

    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'test_prediction.csv')
    prediction =  rf.predict(X_test)
    np.savetxt(outfile, prediction, delimiter=",")

    outfile = os.path.join(ddir, 'RandomForestTests', 'RFA_Outputs', folder_output, 'training_prediction.csv')
    prediction =  rf.predict(X_train)
    np.savetxt(outfile, prediction, delimiter=",")

    # # 3rd step: apply model
    print("\nApplying model to rest of imagery")
    print("model save:", model_save)
    apply_model(img, imgProperties, classDir, model_save)

    # # 4th step: run diagnostics
    # run_diagnostics(model_save, X_test, y_test, fid_test)
    #
    # elapsed = round(find_elapsed_time(start, timer()), 3)
    #
    # print("\n\nElapsed time = {}".format(elapsed), "mins")


### FOR PROJECT: Run the model with different parameters:
##trees = [30, 200]
##feats = ['sqrt', 'log2']
##for n_trees in trees:
##    for max_feat in feats:
##        modelName = '{}_{}'.format(n_trees, max_feat)# 'try3' # to distinguish between parameters of each model
##        main()


if __name__ == '__main__':
    main()
