#!/usr/bin/env python3
'''
Plot GDAL-compatible map data sets, including complex images and multi-band
 data sets.
'''

### IMPORT MODULES ---
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.interpolate import interp1d
from osgeo import gdal


### PARSER ---
def createParser():
    Description = '''Plot GDAL-compatible map data sets, including complex
images and multi-band data sets.'''

    parser=argparse.ArgumentParser(description = Description,
        formatter_class = argparse.RawTextHelpFormatter)

    # Input arguments
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    inputArgs.add_argument(dest='imgFile', type=str, 
        help='File to plot')
    inputArgs.add_argument('-i','--image-type', dest='imgType', type=str, default='auto',
        help='Image type ([auto], ISCE/complex)')
    inputArgs.add_argument('-b','--band', dest='imgBand', type=int, default=1,
        help='Image band')
    inputArgs.add_argument('-bg','--background', dest='background', nargs='+', 
        default=None, help='Background value')

    # Display arguments
    displayArgs = parser.add_argument_group('DISPLAY ARGUMENTS')
    displayArgs.add_argument('-ds','--downsample-factor', dest='dfactor', default=0,
        help='Downsample factor')
    displayArgs.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap')
    displayArgs.add_argument('-co','--cbar-orient', dest='cOrient', type=str,
        default='horizontal', help='Colorbar orientation ([horizontal], vertical')
    displayArgs.add_argument('-vmin','--vmin', dest='vmin', type=float, default=None,
        help='Minimum value to plot (overridden by pctmin)')
    displayArgs.add_argument('-vmax','--vmax', dest='vmax', type=float, default=None,
        help='Maximum value to plot (overridden by pctmax)')
    displayArgs.add_argument('-pctmin','--percent-min', dest='pctmin', type=float, 
        default=None, help='Minimum percent clip')
    displayArgs.add_argument('-pctmax','--percent-max', dest='pctmax', type=float,
        default=None, help='Maximum percent clip')
    displayArgs.add_argument('-eq','--equalize', dest='equalize', action='store_true',
        help='Equalize')

    # Output arugments
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    outputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode')
    outputArgs.add_argument('-hist','--histogram', dest='plotHist', action='store_true',
        help='Plot histogram (background ignored)')
    outputArgs.add_argument('-o','--outName', dest='outName', type=str, default=None,
        help='Output name')
    outputArgs.add_argument('-of','--output-format', dest='outFmt', type=str, default='png',
        help='Output format ([png],pdf)')
    outputArgs.add_argument('--nodisplay', dest='noDisplay', action='store_true',
        help='No call to display plot. Recommend -o to save result.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()

    return parser.parse_args(args = iargs)



### MAPSHOW OBJECT ---
class mapShow:
    '''
    Object for displaying maps.
    '''

    ## Load and format
    def __init__(self, imgFile, imgType, imgBand, background, verbose):
        '''
        Instantiate object: Store parameters; Load map data; Conduct 
         behind-the-scenes formatting.
        '''
        # Record parameters
        self.imgFile = os.path.abspath(imgFile)
        self.imgType = imgType
        self.imgBand = imgBand
        self.background = background
        self.verbose = verbose

        # Load and format data
        self.__loadData__()
        self.__maskBackground__()
        self.__computeImgStats__()


    def __loadData__(self):
        '''
        Load data and format based on input criteria.
        Format geographic extent.
        Called automatically by __init__.
        '''
        # Confirm file exists
        if not os.path.exists(self.imgFile):
            print('File does not exist:\n\t{:s}'.format(self.imgFile))
            exit()
        else:
            if self.verbose == True: print('Loading: {:s}'.format(self.imgFile))

        # Load data
        self.DS = gdal.Open(self.imgFile, gdal.GA_ReadOnly)

        # Image type
        if self.imgType in 'ISCE, complex':
            self.amp = self.DS.GetRasterBand(1).ReadAsArray()
            self.phs = self.DS.GetRasterBand(2).ReadAsArray()

            if self.verbose == True:
                print('Loaded complex image.\nBand 1: Amplitude\nBand 2: Phase')
        else:
            self.img = self.DS.GetRasterBand(self.imgBand).ReadAsArray()

            if self.verbose == True: 
                print('Loaded band: {:d} / {:d}'.format(self.imgBand,
                    self.DS.RasterCount))

        self.imgSize = np.prod(self.img.shape)

        # Format geographic extent
        self.__formatGeographic__()

    def __formatGeographic__(self):
        '''
        Format the geographic extent of an image for plotting purposes.
        Called automatically by __loadData__.
        '''
        # Basic parameters
        M,N = self.DS.RasterYSize, self.DS.RasterXSize
        [xmin,dx,xshear,ymax,yshear,dy] = self.DS.GetGeoTransform()

        # Fill in the rest
        xmax = xmin + N*dx
        ymin = ymax + M*dy

        # Format extent
        self.extent = (xmin, xmax, ymin, ymax)

        # Report if requested
        if self.verbose == True:
            print('GEOGRAPHIC INFO')
            print('Spatial extent:\n\txmin {}\n\tymin {}\n\txmax {}\n\tymax {}'.\
                format(xmin, ymin, xmax, ymax))
            print('Pixel sizes: dx {}, dy {}'.format(dx,dy))

    def __maskBackground__(self):
        '''
        Mask 'img' data based on specified background values.
        Called automatically by __init__.
        '''
        if self.verbose == True: print('BACKGROUND VALUE(S):')
        mask = np.ones(self.img.shape)

        # Mask by background value(s)
        if self.background is not None:
            # Number of background values
            nBackground = len(self.background)
            print('{:d} background values specified'.format(nBackground))

            # Formatted list of background values
            BGvals = []
            for n,val in enumerate(self.background):
                if val == 'auto':
                    # Automatically detect background
                    val = self.__detectBackground__(self.img)
                else:
                    # Use specified background value
                    val = float(val)

                # Append result to array
                BGvals.append(val)

                # Report results of each background value
                if self.verbose == True:
                    pct = np.sum(self.img==val)/self.img.size
                    print('\tValue: {:d} composes {:.1f} %'.format(n,pct))

            # Mask by background value
            mask[np.isnan(self.img) == 1] = 0
            for n,val in enumerate(BGvals):
                mask[self.img == val] = 0

        # If no background values specified
        else:
            if self.verbose == True: print('None')

        # Apply mask
        self.img = np.ma.array(self.img, mask=(mask==0))


    def __detectBackground__(self,img):
        '''
        Use the mode of the image peripheries to automatically detect the
         image background.
        Called automatically by __maskBackground__.
        '''
        # Edge values
        edgeValues = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]])
        autoBG = mode(edgeValues).mode[0]

        return autoBG


    def __computeImgStats__(self):
        '''
        Compute image stats. Ignore background value(s) if provided.
        Called automatically by __init__.
        '''
        # Compute statistics
        self.mean = self.img.compressed().flatten().mean()
        self.median = np.median(self.img.compressed().flatten())
        self.min = self.img.compressed().flatten().min()
        self.max = self.img.compressed().flatten().max()

        # Report if requested
        if self.verbose == True:
            print('IMAGE STATISTICS')
            print('Image shape: {:d} x {:d}'.format(*self.img.shape))
            print('Mean: {:.3e}'.format(self.mean))
            print('Median: {:.3e}'.format(self.median))
            print('Min: {:.3e}'.format(self.min))
            print('Max: {:.3e}'.format(self.max))


    ## Plot
    def plotImg(self,dfactor,cmap,cOrient,vmin,vmax,pctmin,pctmax,equalize):
        '''
        Plot formatted image.
        '''
        # Determine min/max image values
        self.__determineMinMax__(vmin,vmax,pctmin,pctmax)

        # Equalize colors if specified
        if equalize == True: self.__equalize__()

        # Spawn figure
        self.Fig, self.imgAx = plt.subplots()

        # Plot image
        ds = int(2**dfactor)
        cImg = self.imgAx.imshow(self.img[::ds,::ds],
            cmap = cmap, vmin = self.vmin,vmax = self.vmax,
            extent = self.extent)

        # Format image plot
        self.Fig.colorbar(cImg,ax=self.imgAx,orientation=cOrient)

    def __determineMinMax__(self,vmin,vmax,pctmin,pctmax):
        '''
        Determine the image min/max values based on the specifications.
        If multiple values are specified, take conservative min/max.
        Called automatically by plotImg.
        '''
        # Default values
        self.vmin = self.min
        self.vmax = self.max

        # Check for specified values
        if vmin: self.vmin = vmin
        if vmax: self.vmax = vmax

        if pctmin:
            pctmin = np.percentile(self.img.compressed().flatten(),pctmin)
            self.vmin = np.max([self.vmin, pctmin])
        if pctmax:
            pctmax = np.percentile(self.img.compressed().flatten(),pctmax)
            self.vmax = np.min([self.vmax, pctmax])

        # Report min/max value
        if self.verbose == True:
            print('Clipping to min: {:.3f}/max: {:.3f}'.\
                format(self.vmin,self.vmax))

    def __equalize__(self):
        '''
        Equalize color balance for an image.
        '''
        # Store mask for later
        mask = self.img.mask

        # Compute histogram
        hcenters, hvals = self.__computeHistogram__()
        hcenters[0], hcenters[-1] = (self.min, self.max)

        # Integrate to build transform
        hvals = hvals/len(self.img.compressed())
        Hvals = np.cumsum(hvals)
        Hvals[0], Hvals[-1] = (0, 1)

        # Inverse interpolation
        I = interp1d(hcenters,Hvals,
            bounds_error=False)

        # Re-mask image
        self.img = I(self.img.data)
        self.img = np.ma.array(self.img, mask=mask)

        # Replace min/max values
        self.vmin, self.vmax = (0, 1)

    def __computeHistogram__(self):
        '''
        Compute a histogram based on the non-masked image values.
        '''
        # Compute histogram
        hvals, hedges = np.histogram(self.img.compressed().flatten(),bins=128)
        hcenters = (hedges[:-1]+hedges[1:])/2

        return hcenters, hvals


    def plotHistogram(self):
        '''
        Plot histogram of image data (background ignored).
        '''
        # Compute histogram
        hcenters, hvals = self.__computeHistogram__()

        # Spawn figure
        Hist, histAx = plt.subplots()

        # Plot histogram
        markerline, stemlines, baseline = plt.stem(hcenters,hvals,
            linefmt='r',markerfmt='',use_line_collection=True)
        stemlines.set_linewidths(None)
        baseline.set_linewidth(0)
        histAx.plot(hcenters,hvals,'k',linewidth=2)

        # Format histogram
        histAx.set_yticks([])

        histAx.set_xlim([self.vmin,self.vmax])
        histAx.set_xlabel('value')


    ## Save
    def saveMap(self,outName,outFmt):
        '''
        Save the output figure to the given name with the given format.
        '''
        # Construct output name
        outName = '{:s}.{:s}'.format(outName,outFmt)

        # Save figure
        self.Fig.savefig(outName,dpi=600,format=outFmt)

        # Report if requested
        if self.verbose == True: print('Saved to {:s}'.format(outName))



### MAIN ---
if __name__ == '__main__':
    # Gather arguments
    inps = cmdParser()

    # Instantiate object
    M = mapShow(inps.imgFile, inps.imgType, inps.imgBand, inps.background, 
        inps.verbose)

    # Plot image
    M.plotImg(inps.dfactor,
        inps.cmap, inps.cOrient,
        inps.vmin, inps.vmax,
        inps.pctmin, inps.pctmax,
        inps.equalize)

    # Plot histogram
    if inps.plotHist == True: M.plotHistogram()

    # Save if requested
    if inps.outName: M.saveMap(inps.outName,inps.outFmt)

    # Display
    if inps.noDisplay == False: plt.show()