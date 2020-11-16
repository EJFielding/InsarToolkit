#!/usr/bin/env python3
'''
Convert a shapefile to a raster mask
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from osgeo import gdal, ogr


### PARSER ---
Description = '''Convert a shapefile into a raster of equivalent dimensions to a given GeoTiff.

*** The mask returned has values 0 = masked; 1 = valid ***
       This is the opposite of the MODIS convention.
'''

def createParser():
    parser = argparse.ArgumentParser(description=Description)

    # Required inputs
    parser.add_argument('-s','--shapefile', dest='shpName', type=str, required=True,
        help='Shapefile name')
    parser.add_argument('-r','--reference-file', dest='refName', type=str, required=True,
        help='Reference raster name')

    parser.add_argument('-o','--outname', dest='outName', type=str, default='out',
        help='Output filename')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args = iargs)



### FUNCTIONS ---
## Loading
def loadReference(refName):
    '''
    Load and format reference raster dataset using gdal.
    '''
    # Data set
    DS = gdal.Open(refName, gdal.GA_ReadOnly)

    # Check that data set is valid
    if DS is None:
        print('Could not open Reference Raster {:s}'.format(refName))
        exit()
    else:
        print('Reference raster: {:s}'.format(refName))

    # Return
    return DS


def loadShapefile(shpName):
    '''
    Load and format shapefile using ogr.
    '''
    # Define driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Open data set
    DSshp = driver.Open(shpName, 0)

    # Check that data set is valid
    if DSshp is None:
        print('Could not open ESRI Shapefile {:s}'.format(shpName))
        exit()
    else:
        print('Shapefile: {:s}'.format(shpName))

    # Return
    return DSshp


## Geo-formatting
def formatRefExtent(RefDS):
    '''
    Format Pyplot extent given a gdal data set.
    '''
    # Format extent
    tnsf = RefDS.GetGeoTransform()
    M, N = RefDS.RasterYSize, RefDS.RasterXSize
    left,dx,_,top,_,dy = tnsf
    right = left+N*dx
    bottom = top+M*dy

    extent = (left, right, bottom, top)

    # Report
    print('Reference extent:\n\tx: {:.3f} {:.3f}\n\ty: {:.3f} {:.3f}'. \
        format(left, right, bottom, top))

    return extent


## Plotting
def plotReference(Fig,ax,DSref,extent):
    '''
    Plot the reference raster.
    '''
    # Format map image
    img = DSref.GetRasterBand(DSref.RasterCount).ReadAsArray()

    # Determine color coding
    img = maskBackground(img)
    vmin, vmax = np.percentile(img.compressed().flatten(), [1, 99])

    # Plot image
    cax = ax.imshow(img, cmap='jet',
        vmin=vmin, vmax=vmax, extent=extent)

    # Format subplot
    ax.set_aspect(1)

    return Fig, ax

def maskBackground(img):
    '''
    Automatically detect and mask background.
    This is automatically called by the plotReference function.
    '''
    # Convert NaNs to zeros
    img[np.isnan(img) == 1] = 0.

    # Detect background
    edgeValues = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
    BG = mode(edgeValues).mode[0]

    img = np.ma.array(img, mask = (img == BG))

    return img

def plotShapefile(Fig, ax, DSshp, extent):
    '''
    Plot shapefile within given extent.
    '''
    # Format shapefile
    layer = DSshp.GetLayer()
    feature = layer[0]
    geomRef = feature.GetGeometryRef()
    geomXYZ = geomRef.GetGeometryRef(0)
    x = [geomXYZ.GetX(i) for i in range(geomXYZ.GetPointCount())]
    y = [geomXYZ.GetY(i) for i in range(geomXYZ.GetPointCount())]
    z = [geomXYZ.GetZ(i) for i in range(geomXYZ.GetPointCount())]

    # Plot shapefile
    ax.fill(x, y, color='gray')
    ax.plot(x, y, 'k.')

    # Format plot
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect(1)

    return Fig, ax

def plotMask(Fig, ax, DSmsk, extent):
    '''
    Plot rasterized mask.
    '''
    # Format mask image
    msk = DSmsk.GetRasterBand(1).ReadAsArray()

    # Mask reference image
    ax.imshow(msk, extent=extent)

    return Fig, ax

def plotFinal(Fig, ax, DSref, DSmsk, extent):
    '''
    Plot final result: Masked data set.
    '''
    # Format reference image
    img = DSref.GetRasterBand(DSref.RasterCount).ReadAsArray()
    img = maskBackground(img)
    vmin, vmax = np.percentile(img.compressed().flatten(), [1, 99])

    # Format mask image
    msk = DSmsk.GetRasterBand(1).ReadAsArray()

    # Mask reference image
    img = np.ma.array(img, mask = (msk==0))

    # Plot masked image
    cax = ax.imshow(img, cmap='jet',
        vmin=vmin, vmax=vmax, extent=extent)

    return Fig, ax


## Convert to raster
def Shp2Raster(DSref, DSshp, outName):
    '''
    Convert shapefile to raster.
    '''
    # Format output name
    if outName.split('.')[-1] != '.tif':
        outName += '.tif'

    print('Saving to: {:s}'.format(outName))

    # Shapefile layer
    layer = DSshp.GetLayer()

    # Create target data set
    driver = gdal.GetDriverByName('GTiff')
    outDS = driver.Create(outName,DSref.RasterXSize,DSref.RasterYSize,1,
        gdal.GDT_Float32)
    outDS.SetGeoTransform(DSref.GetGeoTransform())
    outDS.SetProjection(DSref.GetProjection())
    gdal.RasterizeLayer(outDS, [1], layer, burn_values=[1])
    band = outDS.GetRasterBand(1)
    mask = band.ReadAsArray()
    mask = -1*mask + 1  # invert
    band.WriteArray(mask)

    outDS.FlushCache()

    return outDS



### MAIN ---
if __name__ == '__main__':
    ## Setup
    # Gather inputs
    inps = cmdParser()

    # Spawn figure
    Fig, [axRef, axShp, axMsk, axFinal] = plt.subplots(figsize = (14,8), ncols = 4)


    ## Reference raster
    DSref = loadReference(inps.refName)
    extent = formatRefExtent(DSref)

    # Plot reference raster
    plotReference(Fig, axRef, DSref, extent)


    ## Shapefile
    # Load shapefile data set
    DSshp = loadShapefile(inps.shpName)

    # Plot shapefile
    plotShapefile(Fig, axShp, DSshp, extent)


    ## Shapefile to raster
    # Convert shapefile to raster
    DSmsk = Shp2Raster(DSref, DSshp, inps.outName)

    # Plot mask
    plotMask(Fig, axMsk, DSmsk, extent)

    # Plot result
    plotFinal(Fig, axFinal, DSref, DSmsk, extent)

    plt.show()