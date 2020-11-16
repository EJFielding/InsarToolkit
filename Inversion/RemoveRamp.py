#!/usr/bin/env python3
'''
Remove a linear ramp from a data set.
'''

### IMPORT MODULES ---
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from osgeo import gdal


### PARSER ---
Description = '''Resample a mask to the same size and resolution as the input interferogram.

*** The mask returned has values 0 = masked; 1 = valid ***
       This is the opposite of the MODIS convention.
'''

def createParser():
    parser = argparse.ArgumentParser(description=Description)

    # Required inputs
    parser.add_argument(dest='fname', type=str,
        help='File name.')

    # Optional inputs
    parser.add_argument('-m','--mask', dest='maskVals', nargs='+',
        default=[], help='Mask values/names')
    parser.add_argument('-c','--coherence', dest='cohName', default=None,
        help='Coherence map for weighting')
    parser.add_argument('-n','--nsamples', dest='nSamples', default=1000,
        help='Number of samples')
    parser.add_argument('--remove-offset', dest='removeOffset', action='store_true',
        help='Remove offset')

    # Output arguments
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose')
    parser.add_argument('-o','--outname', dest='outName', type=str, default='out',
        help='Output filename')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args = iargs)



### CLASS ---
class rampRemoval:
    '''
    Remove ramp from georeferenced data set.
    '''
    def __init__(self,verbose=False):
        '''
        Initialize object. Nothing else needed now.
        '''
        self.verbose = verbose

        if self.verbose == True: print('Initiallzing ramp removal')

    def loadDataset(self,fname):
        '''
        Load input data set in gdal format.
        Automatically format spatial extent and create reference grid.
        Initialize NaN mask.
        '''
        # Load gdal data set
        self.DS = gdal.Open(fname, gdal.GA_ReadOnly)

        # Check that data set loaded properly
        if self.DS is None:
            print('Input data set: {:s} not loaded'.format(fname))
            exit()
        else:
            if self.verbose == True: 
                print('Loaded input data set: {:s}'.format(fname))

        # Store image values
        self.Z = self.DS.GetRasterBand(self.DS.RasterCount).ReadAsArray()

        # Get extent
        self.__getExtent__()

        # Create grid
        self.__createGrid__()

        # Initialize mask
        self.__initMask__()

        # Initialize weights
        self.__initWeights__()

    def __getExtent__(self):
        '''
        Get geographic extent from gdal dataset.
        Called automatically by loadDataset.
        '''
        self.M = self.DS.RasterYSize
        self.N = self.DS.RasterXSize
        left, dx, _, top, _, dy = self.DS.GetGeoTransform()
        right = left + dx*self.N
        bottom = top + dy*self.M

        self.extent = (left, right, bottom, top)

    def __createGrid__(self):
        '''
        Create reference grid.
        Called automatically by loadDataset.
        '''
        left, right, bottom, top = self.extent

        x = np.linspace(left, right, self.N)
        y = np.linspace(top, bottom, self.M)

        self.X, self.Y = np.meshgrid(x, y)

    def __initMask__(self):
        '''
        Initialize mask by masking out NaN values.
        Ones = pass
        Zeros = mask
        Called automatically by loadDataset.
        '''
        # Mask of all ones
        self.mask = np.ones((self.M, self.N))

        # Mask NaNs
        img = self.DS.GetRasterBand(self.DS.RasterCount).ReadAsArray()

        self.mask[np.isnan(img) == 1] = 0

    def __initWeights__(self):
        '''
        Initialize weights as all ones.
        Called automatically by loadDataset.
        '''
        # Weights map of all ones
        self.weights = np.ones((self.M, self.N))


    def maskValues(self,maskVals):
        '''
        Mask by value, map, or automatically detected background.
        '''
        # Load image
        img = self.DS.GetRasterBand(self.DS.RasterCount).ReadAsArray()

        # Loop through maskVals
        for maskVal in maskVals:
            # Masking map
            if os.path.exists(maskVal):
                MSK = gdal.Open(maskVal, gdal.GA_ReadOnly)
                msk = MSK.GetRasterBand(1).ReadAsArray()
                self.mask[msk == 0] = 0

                if self.verbose == True:
                    print('Masking by data set: {:s}'.format(maskVal))
            # Background value
            if maskVal in ['auto', 'bg', 'background']:
                # Auto-detect background value
                self.__detectBackground__(img)
                self.mask[img == self.bg] = 0

                if self.verbose == True:
                    print('Masking background value: {:f}'.format(self.bg))
            # Float value to mask
            try:
                maskVal = float(maskVal)
                self.mask[img == maskVal] = 0

                if self.verbose == True:
                    print('Masking value: {:f}'.format())
            except:
                pass

    def __detectBackground__(self,img):
        '''
        Detect the background value of an image.
        Called automatically by maskValues.
        '''
        edgeValues = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
        self.bg = mode(edgeValues).mode[0]


    def coherenceWeights(self,cohName):
        '''
        Weight by coherence (phase sigma).
        '''
        # Load coherence data set
        COH = gdal.Open(cohName, gdal.GA_ReadOnly)
        coh = COH.GetRasterBand(1).ReadAsArray()

        # Update weights
        self.weights *= coh

        # Report
        if self.verbose == True:
            print('Updating weight map using: {:s}'.format(cohName))


    def removePlane(self, nSamples):
        '''
        Remove a linear ramp (plane).
        '''
        # Draw samples
        self.__drawSamples__(nSamples)

        # Fit plane
        self.__fitPlane__()

        # Subtract plane
        self.__subtractPlane__()

    def __drawSamples__(self,nSamples):
        '''
        Draw evenly spaced samples from valid data points.
        Called automatically by removePlane.
        '''
        # Mask data sets
        X = np.ma.array(self.X, mask = (self.mask == 0))
        Y = np.ma.array(self.Y, mask = (self.mask == 0))
        Z = np.ma.array(self.Z, mask = (self.mask == 0))
        W = np.ma.array(self.weights, mask = (self.mask == 0))

        nPts = len(X.compressed().flatten())  # nb valid points
        skips = int(nPts/nSamples)

        # Sample points
        self.sx = X.compressed().flatten()[::skips]
        self.sy = Y.compressed().flatten()[::skips]
        self.sz = Z.compressed().flatten()[::skips]
        self.sw = W.compressed().flatten()[::skips]

        # Number of samples
        self.nSamples = len(self.sx)

        # Report
        if self.verbose == True:
            print('Drawing {:d} samples'.format(self.nSamples))
            print('Original data points: {:d}'.format(self.M*self.N))
            print('Valid data points:    {:d}'.format(nPts))
            print('Sampling every {:d} valid points'.format(skips))

    def __fitPlane__(self):
        '''
        Fit a plane to the masked data set.
        Called automatically by removePlane.
        '''
        if self.verbose == True: print('Fitting plane')

        # Remove mean values
        self.xmean = self.sx.mean()
        self.ymean = self.sy.mean()
        self.zmean = self.sz.mean()

        # Reshape into 1D arrays
        x = self.sx.reshape(self.nSamples,1)-self.xmean
        y = self.sy.reshape(self.nSamples,1)-self.ymean
        z = self.sz.reshape(self.nSamples,1)-self.zmean

        # Design matrix
        G = np.hstack([x,y])

        # Weights matrix
        W = np.zeros((self.nSamples,self.nSamples))
        for i in range(self.nSamples):
            W[i,i] = self.sw[i]

        # Solve for plane
        self.beta = np.linalg.inv(G.T.dot(W).dot(G)).dot(G.T).dot(W).dot(z)

        # Report
        if self.verbose == True:
            print('Plane parameters:\n\tx {:f}\n\ty {:f}'.\
                format(self.beta[0,0],self.beta[1,0]))

    def __subtractPlane__(self):
        '''
        Subtract plane from original data set.
        Called automatically by removePlane.
        '''
        MN = self.M*self.N

        # Reshape into 1D arrays
        x = self.X.reshape(MN,1)-self.xmean
        y = self.Y.reshape(MN,1)-self.ymean

        # Design matrix
        G = np.hstack([x,y])

        # Compute plane
        P = G.dot(self.beta).reshape(self.M, self.N)

        # Deramped image
        self.deramped = self.Z - P


    def plot(self):
        '''
        Plot maps.
        '''
        # Spawn figure
        self.Fig, [axInputs, axMask, axWeights, axSamples, axDeramp] = \
                plt.subplots(figsize = (14, 8), ncols = 5)

        # Plot input data set
        self.__plotInputs__(axInputs)

        # Plot mask
        self.__plotMask__(axMask)

        # Plot weights
        self.__plotWeights__(axWeights)

        # Plot samples
        self.__plotSamples__(axSamples)

        # Plot result
        self.__plotResult__(axDeramp)

        # Format figure
        self.Fig.tight_layout()

    def __plotInputs__(self, ax):
        '''
        Plot input data set.
        Called automatically by plot.
        '''
        # Mask values
        img = np.ma.array(self.Z, mask = (self.mask == 0))

        # Plot map
        cax = ax.imshow(img, cmap='jet', extent=self.extent)

        # Formatting
        ax.set_title('Orig image')
        ax.set_aspect(1)
        self.Fig.colorbar(cax, ax=ax, orientation='horizontal')

    def __plotMask__(self, ax):
        '''
        Plot mask data set.
        Called automatically by plot.
        '''
        # Plot mask
        cax = ax.imshow(self.mask, extent=self.extent)

        # Formatting
        ax.set_title('Mask')
        ax.set_aspect(1)
        self.Fig.colorbar(cax, ax=ax, orientation='horizontal')

    def __plotWeights__(self, ax):
        '''
        Plot weight map.
        Called automatically by plot.
        '''
        # Mask weights
        weights = np.ma.array(self.weights, mask = (self.mask == 0))

        # Plot weight map
        cax = ax.imshow(weights, extent=self.extent)

        # Formatting
        ax.set_title('Weights')
        ax.set_aspect(1)
        self.Fig.colorbar(cax, ax=ax, orientation='horizontal')

    def __plotSamples__(self, ax):
        '''
        Plot sample points drawn.
        Called automatically by plot.
        '''
        # Plot samples
        cax = ax.scatter(self.sx, self.sy, s=5, c=self.sz, cmap='jet')

        # Format plot
        ax.set_title('Samples')
        ax.set_xlim(self.extent[:2])
        ax.set_ylim(self.extent[2:])
        ax.set_aspect(1)
        self.Fig.colorbar(cax, ax=ax, orientation='horizontal')

    def __plotResult__(self, ax):
        '''
        Plot deramped results.
        Called automatically by plot.
        '''
        # Mask results
        Dimg = np.ma.array(self.deramped, mask = (self.mask == 0))

        # Plot results
        cax = ax.imshow(Dimg, cmap='jet', extent=self.extent)

        # Formatting
        ax.set_title('Deramped image')
        ax.set_aspect(1)
        self.Fig.colorbar(cax, ax=ax, orientation='horizontal')


    def saveResult(self, outName):
        '''
        Save to GeoTiff.
        '''
        Driver = gdal.GetDriverByName('GTiff')
        outDS = Driver.Create(outName,self.N,self.M,1,gdal.GDT_Float32)
        outDS.SetGeoTransform(self.DS.GetGeoTransform())
        outDS.SetProjection(self.DS.GetProjection())
        outDS.GetRasterBand(1).WriteArray(self.deramped)
        outDS.FlushCache()

        if self.verbose == True:
            print('Saved to: {:s}'.format(outName))



### MAIN ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()

    # Instantiate ramp fitting object
    R = rampRemoval(verbose = inps.verbose)

    # Load input data set
    R.loadDataset(inps.fname)

    # Mask by value or map
    R.maskValues(inps.maskVals)

    # Weighting data set
    R.coherenceWeights(inps.cohName)

    # Remove plane
    R.removePlane(nSamples = inps.nSamples)

    # Plot
    R.plot()

    # Save result
    if inps.outName:
        R.saveResult(inps.outName)


    plt.show()