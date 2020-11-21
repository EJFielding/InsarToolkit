#!/usr/bin/env python3
'''
Extract a 1D profile from an image.
Images should ideally be in GDAL georeferenced format.
'''

### IMPORT MODULES ---
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import mode
from osgeo import gdal


### PARSER ---
def createParser():
    Description = '''Extract a 1D profile from an image.
Designed for use with georeferenced images encoded in GDAL format.
'''

    parser=argparse.ArgumentParser(description = Description,
        formatter_class = argparse.RawTextHelpFormatter)

    # Input arguments
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    inputArgs.add_argument(dest='imgFile', type=str, 
        help='File to plot')
    inputArgs.add_argument('-t','--image-type', dest='imgType', type=str, default='auto',
        help='Image type ([auto], ISCE/complex)')
    inputArgs.add_argument('-b','--band', dest='imgBand', type=int, default=1,
        help='Image band')
    inputArgs.add_argument('-n','--nodata', dest='noData', nargs='+', default=[],
        help='No data values')

    # Display arguments
    displayArgs = parser.add_argument_group('DISPLAY ARGUMENTS')
    displayArgs.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap')
    displayArgs.add_argument('-co','--cbar-orient', dest='cbarOrient', type=str,
        default='horizontal', help='Colorbar orientation ([horizontal], vertical')
    displayArgs.add_argument('-bg','--background', dest='background', default=None,
        help='Background value')
    displayArgs.add_argument('-pctmin','--percent-min', dest='pctmin', type=float, 
        default=0, help='Minimum percent clip')
    displayArgs.add_argument('-pctmax','--percent-max', dest='pctmax', type=float,
        default=100, help='Maximum percent clip')

    # Profile arguments
    profileArgs = parser.add_argument_group('PROFILE ARGUMENTS')
    profileArgs.add_argument('-w','--profile-width', dest='profWidth', type=float, default=None,
        help='Profile width in pixels.')
    profileArgs.add_argument('-qXY','--queryXY', dest='queryXY', default=None, nargs=2,
        help='Query point in image XY coordinates.')
    profileArgs.add_argument('-qLoLa','--queryLoLa', dest='queryLoLa', default=None, nargs=2,
        help='Query point in geographic coordinates')
    profileArgs.add_argument('--binning', dest='binning', action='store_true',
        help='Smooth profile values by binning.')
    profileArgs.add_argument('--bin-spacing', dest='binSpacing', type=float, default=None,
        help='Bin spacing in map units')
    profileArgs.add_argument('--bin-widths', dest='binWidths', type=float, default=None,
        help='Bin widths in map units')

    # Output arugments
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    outputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode')
    outputArgs.add_argument('-o','--outname', dest='outName', type=str, default=None,
        help='Output name (no extension). Save points to file.')
    outputArgs.add_argument('--overwrite', dest='overwrite', action='store_true',
        help='Overwrite previous points file')
    outputArgs.add_argument('--profile-start', dest='profStart', type=int, default=1,
        help='Starting number for profile indices')

    return parser

def cmdParser(iargs = None):
    parser = createParser()

    return parser.parse_args(args = iargs)



### PROFILE CLASS ---
class imgProfile:
    '''
    Load an image and collect a profile across it.
    '''
    def __init__(self, imgName, band=1, outName=None):
        '''
        Load and format geographic data set using __loadDS__.
        Format profile width based on explicit input or pixel size using
         __determineProfWidth__.
        '''
        # Instance specs
        self.imgName = imgName
        self.band = band
        self.outName = outName

        # Load image data set
        self.__loadDS__()

        # Image presets
        self.cmap = 'viridis'
        self.cbarOrient = 'horizontal'
        self.background = None
        self.pctmin = 0
        self.pctmax = 100

        # Profile presets
        self.profNb = 1  # profile label number
        self.profWidth = self.pxSize  # initial profile width
        self.binning = False  # for extracting signal from wide profiles
        self.binSpacing = 1*self.pxSize  # pixels
        self.binWidths = 5*self.pxSize  # pixels


    ### Loading
    def __loadDS__(self):
        '''
        Load GDAL-compatible data set.
        Automatically format the image extent using __parseExtent__.
        Forumlate the spatial grids X, Y using __createGrid__.
        Called automatically by __init__.
        '''
        # Load data set
        self.DS = gdal.Open(self.imgName, gdal.GA_ReadOnly)

        # Check validity of data set
        if self.DS is None:
            print('Cannot read {:s} using GDAL.\nCheck path or format'.\
                format(self.imgName))
            exit()
        if self.DS.RasterCount > 0:
            print('Loaded: {:s}'.format(self.imgName))

        # Parse data set properties
        self.__parseExtent__()

        # Create spatial grids
        self.__createGrid__()

    def __parseExtent__(self):
        '''
        Format transform matrix and imshow extent from GDAL geo transform.
        Calculation of right and bottom coordinates relies on __xy2lola__
         function.
        Called automatically by __loadDS__.
        '''
        M, N = self.DS.RasterYSize, self.DS.RasterXSize
        x0,dx,sx,y0,sy,dy = self.DS.GetGeoTransform()

        # Formulate transform matrix
        self.T = np.array([[dx,sy],
                      [sx,dy]])
        self.Tinv = np.linalg.inv(self.T)
        self.l0 = np.array([[x0,y0]]).T

        # Compute right and bottom coordinates using __xy2lola__ transform
        left = x0; top = y0
        right, bottom = self.__xy2lola__(N, M)
        right = right; bottom = bottom

        # Record pixel size
        self.pxSize = np.mean(np.abs([dx, dy]))

        # Formulate imshow extent
        self.extent = (left, right, bottom, top)

        # Report basic parameters
        print('Raster size: {:d} x {:d}'.format(M,N))
        print('Pixel size: {:.5f}'.format(self.pxSize))

    def __xy2lola__(self,x,y):
        '''
        Convert image coordinates to geo coordinates using transform approach.
         l = T * p + l0
         lon  =  |  dx   yshear| * x + lon0
         lat     |xshear   dy  |   y   lat0
        '''
        # Formulate position
        p = np.array([[x,y]]).T

        # Transform
        lon, lat = self.T.dot(p) + self.l0

        return lon[0], lat[0]

    def __lola2xy__(self,lon,lat):
        '''
        Convert geo coordinates to image x,y using transform approach.
         p = Tinv.(l - l0)
         x  =  (|  dx   yshear|)-1 * (lon - lon0)
         y     (|xshear   dy  |)     (lat - lat0)
        '''
        # Formulate geo coordinates
        l = np.array([[lon,lat]]).T

        # Transform
        x, y = self.Tinv.dot((l-self.l0))

        return x[0], y[0]

    def __createGrid__(self):
        '''
        Formulate spatial grids X, Y based on the given image coordinates.
        Called automatically by __loadDS__.
        '''
        left, right, bottom, top = self.extent

        x = np.linspace(left, right, self.DS.RasterXSize)
        y = np.linspace(top, bottom, self.DS.RasterYSize)

        self.X, self.Y = np.meshgrid(x, y)


    ### Plotting
    def showImage(self):
        '''
        Show image with formatting specifications. Get image parameters using
         __imageSpecs__. Display actual image with reuseable __plotImg__ 
         function.
        '''
        # Spawn image figure
        self.ImgFig, self.axImg = plt.subplots(figsize=(8,8))

        # Get specifications
        self.__imageSpecs__()

        # Plot image
        cImg = self.__plotImg__()

        # Format colorbar
        self.ImgFig.colorbar(cImg, ax=self.axImg, orientation=self.cbarOrient)

        # Interact with image
        self.ImgFig.canvas.mpl_connect('button_press_event',
            self.__clickProfile__)

        # Spawn profile fig
        self.ProfFig = plt.figure(figsize=(9,5))

        # Data display axis
        self.axProf = self.ProfFig.add_subplot(position=(0.1, 0.3, 0.8, 0.6))

        # Save profile button 
        self.axSave = self.ProfFig.add_subplot(position=(0.15, 0.05, 0.1, 0.05))
        self.saveButton = Button(self.axSave, 'Save profile', hovercolor='1')
        self.saveButton.on_clicked(self.__saveProfile__)

        # Create profile width slider
        self.axPW = self.ProfFig.add_subplot(position=(0.2, 0.15, 0.6, 0.05))
        self.PWslider = Slider(self.axPW, 'Profile width', 
            self.pxSize, 100*self.pxSize,
            valinit = self.profWidth, valstep = self.pxSize)

        self.PWslider.on_changed(self.__updateProfWidth__)


    def __imageSpecs__(self):
        '''
        Gather image specifications including background, pctmin, pctmax.
        '''
        # Load image as array
        img = self.DS.GetRasterBand(self.band).ReadAsArray()

        # Replace nans
        img[np.isnan(img) == 1] = 0

        # Image background
        if self.background == 'auto':
            # Automatically detect background and mask image
            edgeValues = np.concatenate([img[0,:],img[-1,:],
                img[:,0],img[:,-1]])
            bg = mode(edgeValues).mode[0]
        else:
            try:
                bg = float(self.background)
            except:
                bg = None

        # Carry image as masked array
        self.img = np.ma.array(img, mask=(img==bg))

        # Min/max values
        self.vmin, self.vmax = np.percentile(self.img.compressed().flatten(),
            [self.pctmin, self.pctmax])

    def __plotImg__(self):
        '''
        Plot and format the base map.
        '''
        # Plot image
        cImg = self.axImg.imshow(self.img, 
            cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
            extent=self.extent,
            zorder=1)

        # Format image
        self.axImg.set_aspect(1)

        return cImg

    def __plotStart__(self):
        '''
        Plot profile starting location on image.
        '''
        # Clear current axis
        self.axImg.cla()

        # Replot image
        self.__plotImg__()

        # Plot starting point
        self.axImg.plot(self.x0, self.y0, 'ks')

        # Render image
        self.ImgFig.canvas.draw()

    def __plotProfile__(self):
        '''
        Plot the 1D spatial representation of a profile on the image.
        Then plot the 2D footprint of the profile. Helpful for visualizing 
         multi-pixel profile widths.
        '''
        # Plot starting point
        self.__plotStart__()

        # Plot profile
        self.axImg.plot([self.x0, self.x1],[self.y0, self.y1], 'k', zorder=3)

        # Calculate unit vector
        v,_ = self.__computeProfVector__(self.x0, self.y0, self.x1, self.y1)
        vperp = np.array([v[1],-v[0]])  # perpendicular to profile
        W2 = self.profWidth/2  # half-profile width

        # Compute corner locations
        start = np.array([self.x0, self.y0])
        end = np.array([self.x1, self.y1])
        c1 = W2*vperp + start
        c2 = -W2*vperp + start
        c3 = -W2*vperp + end
        c4 = W2*vperp + end
        
        # Order points for fill plotting
        C = np.array([c1, c2, c3, c4])

        # Plot footprint
        self.axImg.fill(C[:,0], C[:,1], color=(0.6,0.6,0.6), alpha=0.6, zorder=2)

        # Render image
        self.ImgFig.canvas.draw()

    def __plotProfData__(self):
        '''
        Plot 1D profile data and perform basic analysis.
        '''
        # Clear axis
        self.axProf.cla()

        # Plot data points
        self.axProf.plot(self.profDist, self.profPts, linewidth=0, marker='.',
            color=(0.6,0.6,0.6))

        # Smooth using binning
        if self.binning == True:
            xProf, yProf = self.__binning__(self.profDist, self.profPts)
            self.axProf.plot(xProf, yProf, 'b')

        # Format chart
        profTitle = 'Profile {:d} - {:d} data points '.\
                    format(self.profNb,len(self.profDist))
        self.axProf.set_title(profTitle)

        # Render data
        self.ProfFig.canvas.draw()



    ### Profiling
    def queryProfile(self,queryXY=None, queryLoLa=None):
        '''
        Wrapper for computing the profile points.
        '''
        print('Computing profile based on inputs...')
        if queryLoLa:
            print('Initial geographic query points provided')

            # Format query points
            LoLa = [float(l) for lola in queryLoLa for l in lola.split(',')]

        elif queryXY:
            print('Initial XY query points provided')

            # Format query points
            XY = [int(p) for xy in queryXY for p in xy.split(',')]

            # Convert query points to Lon/Lat
            LoLa = []
            LoLa.extend(self.__xy2lola__(*XY[:2]))  # starting position
            LoLa.extend(self.__xy2lola__(*XY[2:]))  # ending position

        # Assign pixel values to object
        self.x0, self.y0, self.x1, self.y1 = LoLa

        # Print stats
        self.__printStart__()
        self.__printEnd__()

        # Plot profile
        self.__plotStart__()
        self.__plotProfile__()

        # Get profile points
        profDist, profPts = self.__generateProfile__()

        # Plot profile
        self.__plotProfData__(profDist, profPts)

    def __clickProfile__(self,event):
        '''
        Get a profile based on user clicks on the plot.
        '''
        # Record click values and take appropriate actions
        if event.button.value == 1:
            self.x0 = event.xdata
            self.y0 = event.ydata
            self.__printStart__()

            # Plot starting location
            self.__plotStart__()

        elif event.button.value == 3:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.__printEnd__()

            # Plot profile line
            self.__plotProfile__()

            # Generate profile
            profDist, profPts = self.__generateProfile__()

    def __printStart__(self):
        '''
        Print the starting coordinates in image and geo coordinates.
        '''
        print('Start')
        print('\tx0 {:.0f}, y0 {:.0f}'.\
            format(*self.__lola2xy__(self.x0, self.y0)))
        print('\tlon0 {:.4f}, lat0 {:.4f}'.format(self.x0, self.y0))

    def __printEnd__(self):
        '''
        Print the ending coordinates in image and geo coordinates.
        '''
        print('End')
        print('\tx1 {:.0f}, y1 {:.0f}'.\
            format(*self.__lola2xy__(self.x1, self.y1)))
        print('\tlon1 {:.4f}, lat1 {:.4f}'.format(self.x1, self.y1))

    def __generateProfile__(self):
        '''
        Get a profile along the given track and with the specified width by
         first rotating the data, then extracting values along the profile.
         Image coordinates are specified as object attributes using the 
         __clickProfile__ or queryProfile fuctions.
        General workflow:
            Formulate profile vector
            Establish rotation matrix
            Center data at profile start and rotate coordinates
            Select points within profile bounds
            Calculate distance along profile from start
        '''
        print('*'*32)
        print('Retreiving profile...')

        # Get map sizes
        M, N = self.DS.RasterYSize, self.DS.RasterXSize
        MN = M*N

        # Get vector showing the direction and length of the profile
        v, self.profLen = self.__computeProfVector__(self.x0,self.y0,
                                                        self.x1,self.y1)
        # Rotation matrix from unit vector
        theta = -np.arctan2(v[1],v[0])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

        # Recenter data at profile start location
        X = self.X - self.x0
        Y = self.Y - self.y0

        # Shape coordinate points into 2 x MN array for rotation operation
        C = np.vstack([X.reshape(1, MN),
                       Y.reshape(1, MN)])
        del X, Y

        # Rotate coordinates
        #  Use abs value of Y coordinates to measure distance from profile
        C = R.dot(C)
        X = C[0,:].reshape(M,N)  # Distance along profile (position)
        Y = np.abs(C[1,:].reshape(M,N))  # Distance from profile (width)
        del C

        # Which points are within profile bounds
        mask = np.ones((M,N))  # start with all valid, then whittle down
        mask[X < 0] = 0  # after profile start
        mask[X > self.profLen] = 0  # ... but before profile end
        mask[Y > self.profWidth] = 0  # within width of profile

        # Retreive points within profile bounds
        profPts = self.img[mask==1]  # image values
        Npts = np.product(profPts.shape)

        # Distance along profile - coordinates already rotated, 
        #  no need for projection 
        profDist = X[mask==1].flatten()

        # Filter for only valid values
        m = profPts.mask
        self.profDist = profDist[m==False]
        self.profPts = profPts[m==False]

        # Print profile properties
        print('\tProfile unit vector: {:.3f}, {:.3f}'.format(*v))
        print('\tProfile width: {:.1f}'.format(self.profWidth))
        print('\tProfile length: {:.1f}'.format(self.profLen))
        print('\tPointing azimuth: {:.1f} deg'.format(90-(180/np.pi)*theta))
        print('\t{:d} points within profile'.format(Npts))

        # Plot profile
        self.__plotProfData__()

        # Return values
        del X, Y
        return profDist, profPts

    def __computeProfVector__(self, x0, y0, x1, y1):
        '''
        Compute the unit vector and length of the profile based on points
         specified using the __clickProfile__ and queryProfile functions.
        '''
        
        v = np.array([x1-x0, y1-y0])
        Len = np.linalg.norm(v)  # length of pointing vector
        v = v/Len

        return v, Len

    def __updateProfWidth__(self,val):
        '''
        Update profile width using slider.
        '''
        # Update value
        self.profWidth = self.PWslider.val

        # Report
        print('*'*32)
        print('Reset profile width to: {:f}'.format(self.profWidth))

        # Recompute profile if already created
        if hasattr(self,'x0'):
            # Replot profile width
            self.__plotProfile__()

            # Recompute profile
            self.__generateProfile__()


    ### Data processing
    def __binning__(self, profDist, profPts):
        '''
        Find moving average of unevenly sampled data track.
        '''
        # Setup
        w2 = self.binWidths/2  # half-width of bins
        x = np.arange(w2,profDist.max()-w2,self.binSpacing)  # distance along profile
        nBins = len(x)  # number of bins
        binStarts = x - w2  # starting point of each bin
        binEnds = x + w2  # ending point of each bin
        yave = np.empty(nBins)
        yave[:] = np.nan

        # Find points in each bin
        for n in range(nBins):
            binPts = profPts[(profDist>=binStarts[n]) & (profDist<binEnds[n])]
            if len(binPts) > 0:
                yave[n] = np.nanmean(binPts)

        return x, yave


    ### Saving
    def __saveProfile__(self,event):
        '''
        Save profile location and data points to a text file.
        '''
        # If outName is specified, save (append) data to file
        if self.outName:
            # Check outName formatting
            if self.outName[-4:] != '.txt': self.outName += '.txt'

            # Write data to file
            with open(self.outName,'a+') as outFile:
                # Format meta information
                metaStr='''---
prof {profNb:d}
x0 {x0:.6f}
y0 {y0:.6f}
x1 {x1:.6f}
y1 {y1:.6f}
width {width:.6f}
n {nPts:d}
'''
                metaDict={'profNb':self.profNb,
                'x0':self.x0, 'y0':self.y0,
                'x1':self.x1, 'y1':self.y1,
                'width':self.profWidth,
                'nPts':len(self.profDist)
                }

                # Write metadata
                outFile.write(metaStr.format(**metaDict))

                # Write profile distances
                distStr = self.__array2str__(self.profDist)
                outFile.write('dist\n')
                outFile.write(distStr+'\n')

                # Write profile data
                ptsStr = self.__array2str__(self.profPts)
                outFile.write('value\n')
                outFile.write(ptsStr+'\n')

            print('Saved to: {:s}'.format(self.outName))

            # Update profile number
            self.profNb += 1

        # If outName not specified, alert user
        else:
            print('No output name specified. Profile not saved!')

    def __array2str__(self,arr):
        '''
        Convert numpy array to string of format 1,2,3,4,5,...,n
        '''
        arr = str([d for d in arr])
        arr = arr.replace('[','')
        arr = arr.replace(']','')
        arr = arr[:-1]

        return arr


### MAIN ---
if __name__ == '__main__':
    # Gather arguments
    inps = cmdParser()

    # Format outName if provided
    if inps.outName:
        # Add .txt extension if not already there
        if inps.outName[-4:] != '.txt': inps.outName+='.txt'

        # Remove old file if specified
        if inps.overwrite == True and os.path.exists(inps.outName):
            os.remove(inps.outName)

    # Load image data set
    prof = imgProfile(inps.imgFile, band=inps.imgBand, outName=inps.outName)

    # Adjust image presets
    prof.cmap = inps.cmap
    prof.cbarOrient = inps.cbarOrient
    prof.background = inps.background
    prof.pctmin = inps.pctmin
    prof.pctmax = inps.pctmax

    # Adjust profile presets
    prof.profNb = inps.profStart
    if inps.profWidth is not None: prof.profWidth = float(inps.profWidth)
    prof.binning = inps.binning
    if inps.binSpacing is not None: prof.binSpacing = inps.binSpacing
    if inps.binWidths is not None: prof.binWidths = inps.binWidths

    # Plot image
    prof.showImage()

    # Compute pre-defined profile
    if inps.queryXY is not None or inps.queryLoLa is not None:
        prof.queryProfile(inps.queryXY, inps.queryLoLa)


    plt.show()