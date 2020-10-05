#!/usr/bin/env python3
'''
Resample a mask to the same size and resolution as the input file.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr


### PARSER ---
Description = '''Resample a mask to the same size and resolution as the input interferogram.

*** The mask returned has values 0 = masked; 1 = valid ***
       This is the opposite of the MODIS convention.
'''

def createParser():
    parser = argparse.ArgumentParser(description=Description)

    # Required inputs
    parser.add_argument('-m','--mask', dest='maskName', type=str, required=True,
        help='Mask name')
    parser.add_argument('-f','--ifg', dest='ifgName', type=str, required=True,
        help='IFG name')

    parser.add_argument('-o','--outname', dest='outName', type=str, default='out',
        help='Output filename')

    # Optional arguments
    parser.add_argument('--mask-value', dest='maskValue', type=str, default=1,
        help='Value that masks data points [default: 1 (MODIS)]')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args = iargs)



### FUNCTIONS ---
## Generic
def loadDS(fname):
    '''
        Load gdal data set and check that it is valid.
    '''
    # Load data set
    DS = gdal.Open(fname,gdal.GA_ReadOnly)

    # Check that the data set is valid
    try:
        DS.GetRasterBand(1).ReadAsArray()
    except:
        print('Not a valid raster data set')
        exit()

    # Report
    print('*'*9)
    print('Loaded data set: {:s}'.format(fname))

    return DS


## Reference data set
def getParameters(DS):
    '''
        Format the parameters necessary to do use gdalwarp
        INPUTS
            DS is the IFG gdal data set
        OUTPUTS
            t_srs is the target projection
            te is the target extent [xmin ymin xmax ymax]
            tr is the target resolution
    '''
    # Format projection
    proj = DS.GetProjection()
    srs = osr.SpatialReference(proj)
    COORDauth = (srs.GetAttrValue("AUTHORITY", 0))
    COORDcode = (srs.GetAttrValue("AUTHORITY", 1))
    t_srs = '{:s}:{:s}'.format(COORDauth,COORDcode)

    # Format extent
    M = DS.RasterYSize
    N = DS.RasterXSize

    tnsf = DS.GetGeoTransform()
    [xmin,dx,xshear,ymax,yshear,dy] = tnsf
    xmax = xmin + dx*N
    ymin = ymax + dy*M

    te = [xmin, ymin, xmax, ymax]
    tr = [dx, dy]

    # Print parameters
    print('*'*9)
    print('Reference parameters:')
    print('Shape: {:d} x {:d}'.format(M,N))
    print('t_srs: {:s}'.format(t_srs))
    print('xmin: {:.5f} ymin: {:.5f} xmax: {:.5f} ymax: {:.5f}'.format(*te))
    print('xres: {:.6f} yres: {:.6f}'.format(*tr))

    # Return parameters
    return t_srs, te, tr


## Mask data set
def transformMask(maskDS,maskValue,t_srs,te,tr,outName):
    '''
        Use gdalwarp to transform the mask data set to the projection and 
         spatial extent of the reference data set.
    '''
    # Format option string
    optionStr = '-t_srs {} -te {} {} {} {} -tr {} {} -r near'.\
        format(t_srs,*te,*tr)

    print('*'*9)
    print('Warping:')
    print(optionStr)

    # Apply transformation
    maskDS = gdal.Warp(outName,maskDS,options=gdal.WarpOptions(format='MEM', 
        dstSRS=t_srs,outputBounds=te,xRes=tr[0],yRes=tr[1],
        resampleAlg='near'))

    # Overwrite data set if mask value is not zero
    if maskValue != 0:
        reformatMask(outName,maskDS,maskValue)

        print('Mask value switched.')
        maskDS = loadDS(outName)

    print('Transformed mask data set saved to: {:s}'.format(outName))

    return maskDS


def reformatMask(outName,DS,maskValue):
    '''
        If the mask value is not zero, rewrite the raster image with the valid
         values as 1. and the masked values as 0.
    '''
    # Original image
    img = DS.GetRasterBand(1).ReadAsArray()

    # Masked pixels in current raster
    msk = (img == maskValue)

    # Rewrite raster
    img = np.ones(img.shape)
    img[msk] = 0

    # Save new raster
    saveDS(outName,img,DS)


def saveDS(outName,img,DS):
    '''
        Save the image to a pre-defined data set.
    '''
    Driver=gdal.GetDriverByName('GTiff')
    OutDS=Driver.Create(outName,DS.RasterXSize,DS.RasterYSize,1,gdal.GDT_Int16) 
    OutDS.GetRasterBand(1).WriteArray(img) 
    OutDS.SetProjection(DS.GetProjection()) 
    OutDS.SetGeoTransform(DS.GetGeoTransform()) 
    OutDS.FlushCache()


## Plotting
def plotDatasets(ifgDS,maskDS):
    '''
        Plot the inputs and results side by side.
    '''
    # Spawn figure
    Fig, [axIFG,axMask,axMaskedIFG] = plt.subplots(ncols=3)

    # Plot original IFG
    Fig, axIFG = plotMap(Fig, axIFG, ifgDS, noData=0)

    # Plot resampled mask
    Fig, axMask = plotMap(Fig, axMask, maskDS)

    # Plot masked IFG
    ifg = ifgDS.GetRasterBand(ifgDS.RasterCount).ReadAsArray()
    msk = maskDS.GetRasterBand(1).ReadAsArray()
    maskedImg = np.ma.array(ifg,mask=((msk==0) | (ifg==0)))

    axMaskedIFG.imshow(maskedImg)
    axMaskedIFG.set_xticks([]); axMaskedIFG.set_yticks([])


def plotMap(Fig, ax, DS, noData=None):
    '''
        Plot the data set in map form.
    '''
    # Get extent from transform
    M = DS.RasterYSize
    N = DS.RasterXSize

    tnsf = DS.GetGeoTransform()
    [left,dx,_,top,_,dy] = tnsf
    right = left + dx*N
    bottom = top + dy*M
    extent = (left, right, bottom, top)

    # Get raster
    nRasters = DS.RasterCount
    img = DS.GetRasterBand(nRasters).ReadAsArray()

    # Apply no data
    img = np.ma.array(img,mask=(img==noData))

    # Plot figure
    cax = ax.imshow(img,extent=extent)

    # Format figure
    Fig.colorbar(cax,ax=ax,orientation='horizontal')

    return Fig, ax



### MAIN ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()

    # Format output name
    if inps.outName.split('.')[-1] != '.tif':
        inps.outName += '.tif'

    # Format parameters for resampling
    ifgDS = loadDS(inps.ifgName)  # load IFG
    t_srs, te, tr = getParameters(ifgDS)

    # Transform mask data set
    maskDS = loadDS(inps.maskName)  # load mask
    maskDS = transformMask(maskDS, inps.maskValue, t_srs, te, tr, inps.outName)

    # Plot results
    plotDatasets(ifgDS,maskDS)


    plt.show()