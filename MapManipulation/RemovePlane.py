#!/usr/bin/env python3
"""
Remove a plane (linear ramp in x,y) from a georeferenced image
"""

### IMPORT MODULES ---
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from osgeo import gdal


### PARSER ---
def createParser():
    parser = argparse.ArgumentParser(description='Remove a plane (linear ramp in x,y) from a georeferenced image')

    # Input data sets
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    parser.add_argument(dest='mapName', type=str,
        help='Map data set from which to remove plane')
    parser.add_argument('-m','--mask', dest='maskValues', nargs='+', default=[],
        help='Mask data set or values (auto for background)')


    # Plane fit parameters
    planeArguments = parser.add_argument_group('PLANE PARAMETERS')
    planeArguments.add_argument('--remove-constant', dest='removeDC', action='store_true',
        help='Remove constant offset')


    # Outputs
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode')

    parser.add_argument('-o','--outName', dest='outName', type=str, default=None,
        help='Output name, for difference map and analysis plots')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## GDAL transform to extent
def transformExtent(DS):
    '''
    Convert GDAL data set to pyplot extent.
    '''
    M = DS.RasterYSize
    N = DS.RasterXSize
    left,dx,_,top,_,dy = DS.GetGeoTransform()
    right = left + N*dx
    bottom = top + M*dy
    extent = (left, right, bottom, top)

    return extent


## Load georeferenced map data set
def createGrid(DS,verbose=False):
    # Geotransform and spatial parameters
    M = DS.RasterYSize
    N = DS.RasterXSize
    xMin,dx,_,yMax,_,dy = DS.GetGeoTransform()
    xMax = xMin + N*dx
    yMin = yMax + M*dy

    x = np.linspace(xMin, xMax, N)
    y = np.linspace(yMin, yMax, M)

    X, Y = np.meshgrid(x, y)

    if verbose == True:
        print('Creating grid')
        print('\tLon: {:f} - {:f}; Lat: {:f} - {:f}'.\
            format(xMin,xMax,yMin,yMax))
        print('\tGrid size: {:d} x {:d}'.format(*X.shape))

    return X, Y


## Mask data set
def maskImg(img,maskValues,verbose=False):
    '''
    Build a mask based on the input image size, given mask values, and/or
     an input feature.
    '''
    # Setup
    mask = np.ones(img.shape)

    # Loop through mask values
    for maskVal in maskValues:
        # Auto detect background value
        if maskVal in ['auto','bg','background']:
            # Detect edge values
            edgeValues = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
            bgVal = mode(edgeValues).mode[0]

            # Update mask
            mask[img == bgVal] = 0

            if verbose == True:
                print('Detecting background value: {:f}'.format(bgVal))
        # Mask by shapefile
        elif os.path.exists(maskVal):
            if verbose == True:
                print('Using mask file: {}'.format(maskVal))
        # Mask by specified value
        else:
            try:
                float(maskVal)
                if verbose == True: print('Mask value: {}'.format(maskVal))
            except:
                print('Mask value {} not recognized -- check path or format'.\
                    format(maskVal))
                exit()

    # Apply mask
    img = np.ma.array(img, mask=(mask==0))

    return img


## Plot map
def plotMap(Fig,ax,img,cmap,extent):
    '''
    Plot georeferenced map.
    '''
    ax.imshow(img,cmap=cmap,extent=extent)

    return Fig, ax


## Save georeferenced data set
def saveMap(inpt,DS,dtrImg):
    # Construct savename
    savename='{}.tif'.format(inpt.outName)

    # GeoTiff
    driver=gdal.GetDriverByName('GTiff')
    DSout=driver.Create(savename,DS.RasterXSize,DS.RasterYSize,1,gdal.GDT_Float32)
    DSout.GetRasterBand(1).WriteArray(dtrImg)
    DSout.SetProjection(DS.GetProjection())
    DSout.SetGeoTransform(DS.GetGeoTransform())
    DSout.FlushCache()



### PLANE FITTING FUNCTIONS ---
## Fit plane
def fitPlane(X,Y,Z,dsFactor,verbose=False):
    '''
    Fit plane to data set.
    '''
    ds = int(2**dsFactor)

    # Number of points to fit
    Xpts = X[::ds,::ds]
    Ypts = Y[::ds,::ds]
    Zpts = Z[::ds,::ds]
    M,N = Zpts.shape
    nPts = M*N

    if verbose == True:
        print('Fitting plane to {:d} points'.format(nPts))

## Fit a plane by linear inversion
# def linearPlaneFit(inpt,X,Y,Z):
#     '''
#     Linear inversion for model:
#      ax + by = z
    
#     Suit data by centering at x = 0, y = 0 and remove mean
#     '''

#     ## Format data
#     # Downsample data
#     # dsFactor=int(2**downsampleFactor)

#     # Format as 1d arrays
#     Xsamp=X.reshape(inpt.M*inpt.N,1)
#     Ysamp=Y.reshape(inpt.M*inpt.N,1)
#     Zsamp=Z.reshape(inpt.M*inpt.N,1)

#     # Discount masked values if applicable
#     if inpt.mask is not None:
#         Xsamp=Xsamp.compressed()
#         Ysamp=Ysamp.compressed()
#         Zsamp=Zsamp.compressed()


#     ## Invert for parameters
#     # Length of data arrays
#     Lsamp=len(Zsamp)

#     # Design matrix
#     G=np.hstack([Xsamp.reshape(Lsamp,1),
#         Ysamp.reshape(Lsamp,1),
#         np.ones((Lsamp,1))])

#     # Parameter solution
#     beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(Zsamp)

#     # Report if requested
#     if inpt.verbose is True:
#         print('Solved for plane.')
#         print('Fit parameters: {}'.format(beta))


#     ## Construct plane
#     # Full design matrix
#     F=np.hstack([X.reshape(inpt.M*inpt.N,1),
#         Y.reshape(inpt.M*inpt.N,1),
#         np.ones((inpt.M*inpt.N,1))])

#     # Plane
#     P=F.dot(beta)
#     P=P.reshape(inpt.M,inpt.N)

#     return P



### MAIN ---
if __name__=='__main__':
    # Gather arguments
    inps = cmdParser()

    # Load map data set
    DS = gdal.Open(inps.mapName,gdal.GA_ReadOnly)
    img = DS.GetRasterBand(1).ReadAsArray()
    extent = transformExtent(DS)

    # Create XY grids
    X,Y = createGrid(DS,verbose=inps.verbose)

    # Mask image
    img = maskImg(img,inps.maskValues,verbose=inps.verbose)

    # Spawn figure
    Fig, [axO, axR, axDR] = plt.subplots(ncols = 3)

    # Plot original image
    plotMap(Fig, axO, img, 'viridis', extent)

    # Fit plane parameters
    fitPlane()

    # Build plane
    buildPlane()

    plt.show(); exit()


    ## Remove plane
    # Fit plane
    P=linearPlaneFit(inpt,X,Y,img)

    # Remove plane
    dtrImg=img-P

    # Plot plane removed
    mapPlot(dtrImg,cmap='viridis',pctmin=inpt.pctmin,pctmax=inpt.pctmax,background=None,
        extent=inpt.T.extent,showExtent=True,cbar_orientation='horizontal',
        title='Plane removed')


    ## Save data set
    if inpt.outName:
        saveMap(inpt,DS,dtrImg)


    plt.show(); exit()