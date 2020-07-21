#!/usr/bin/env python3
'''
Display Relax outputs simply by pointing to the folder.
'''

### IMPORT MODULES ---
import argparse
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


### PARSER ---
Description = '''Plot outputs of Relax model.
'''

Examples = ''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)
    parser.add_argument(dest='fldr', type=str,
        help='Folder with Relax outputs')
    parser.add_argument('-d','--displacement-type', dest='dspType', type=str,
        default='cumulative', help='Displacement type ([cumulative]/postseismic)')
    parser.add_argument('-c','--cmap', dest='cmap', type=str, default='RdBu_r',
        help='Color map')
    parser.add_argument('-p','--palette', dest='palette', nargs = 2, type=float, default=[None,None],
        help='Palette color scale (e.g., -0.05 0.05)')
    parser.add_argument('-s','--vector-steps', dest='steps', type=int, default=3,
        help='Steps (pixels) of horizontal displacement vectors')
    parser.add_argument('-vs','--vector-scale', dest='vscale', type=float, default=1,
        help='Vector scale')
    parser.add_argument('-l','--limits', dest='limits', nargs=4, type=float, default=None,
        help='Plot limits [-30,30,-30,30]')
    parser.add_argument('-los','--los', dest='LOS', action='store_true',
        help='Project into Line of Sight (LOS)')
    parser.add_argument('--alpha', dest='alpha', type=float,
        help='Azimuth angle in degrees CCW between East and the look direction from the target to the sensor. Use 191 for ascending or 169 for descending.')
    parser.add_argument('--theta', dest='theta', type=float,
        help='Incidence angle w.r.t. vertical')
    parser.add_argument('-w','--wrap', dest='wrap', type=float, default=None,
        help='Wrap signal to modulo')
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true',
        default='Verbose mode')
    return parser

def cmdParser(inpt_args=None):
    parser = createParser()
    return parser.parse_args(args=inpt_args)



### DETECT DATA FILES ---
def detectFiles(fldr,dspType='cumulative',verbose=False):
    '''
        Detect relevant files in specified folder.
        Relies on sortFiles function.
    '''
    # Construct search strings
    eastSrchStr = os.path.join(fldr,'*-east.grd')
    northSrchStr = os.path.join(fldr,'*-north.grd')
    upSrchStr = os.path.join(fldr,'*-up.grd')

    # Find all grid files in folder
    eastFiles = glob(eastSrchStr); eastFiles.sort()
    northFiles = glob(northSrchStr); northFiles.sort()
    upFiles = glob(upSrchStr); upFiles.sort()

    # Sort files based on cumulative or postseismic representation.
    eastFiles, northFiles, upFiles = sortFiles(eastFiles, northFiles, upFiles, dspType)

    # Report if requested
    if verbose == True:
        print('{} displacement files'.format(dspType.upper()))
        print('East files:',eastFiles)
        print('North files:',northFiles)
        print('Up files:',upFiles)

    return eastFiles, northFiles, upFiles

def sortFiles(eastFiles,northFiles,upFiles,dspType):
    '''
        Filter files based on displacement type (cumulative/postseismic).
        Part of loadData function.
    '''
    # Number of occurrences of "-relax"
    if dspType == 'cumulative':
        relaxOccurrence = 0
    elif dspType == 'postseismic':
        relaxOccurrence = 1

    eastFiles = [eastFile for eastFile in eastFiles if \
                    eastFile.count('-relax') == relaxOccurrence]
    northFiles = [northFile for northFile in northFiles if \
                    northFile.count('-relax') == relaxOccurrence]
    upFiles = [upFile for upFile in upFiles if \
                    upFile.count('-relax') == relaxOccurrence]

    return eastFiles, northFiles, upFiles



### DISPLACEMENT DATA SET ---
class displacementDataset:
    def __init__(self,name,eastFiles,northFiles,upFiles,
        dataType='cumulative',verbose=False):
        '''
            Initialize.
        '''
        self.name = name
        self.dataType = dataType
        self.verbose = verbose

        # Load data from files
        self.Ueast = self.__loadDspMaps__(eastFiles)
        self.Unorth = self.__loadDspMaps__(northFiles)
        self.Uup = self.__loadDspMaps__(upFiles)

        # Number of epochs to plot
        self.Nepochs = len(eastFiles)

        # Establsih spatial extent
        self.__estbSpatialExtent__(eastFiles[0])

    def __loadDspMaps__(self,filenames):
        '''
            Load grd data using gdal.
        '''
        dspMaps = []
        for filename in filenames:
            # Load grd data
            DS = gdal.Open(filename,gdal.GA_ReadOnly)
            U = DS.GetRasterBand(1).ReadAsArray()
            dspMaps.append(U)
            del DS
        dspMaps = np.array(dspMaps)

        return dspMaps

    def __estbSpatialExtent__(self,filename):
        '''
            Provide a representative data set to establish the spatial
             parameters.
        '''
        # Inherent parameters
        DS = gdal.Open(filename,gdal.GA_ReadOnly)
        self.M = DS.RasterXSize
        self.N = DS.RasterYSize
        tnsf = DS.GetGeoTransform()

        # Spatial extent
        left = tnsf[0]; dx = tnsf[1]; right = left+dx*self.N
        top = tnsf[3]; dy = tnsf[5]; bottom = top+dy*self.M

        self.extent = (left, right, bottom, top)

        # Spatial grid
        x = np.linspace(left,right,self.N)
        y = np.linspace(top,bottom,self.M)

        self.X, self.Y = np.meshgrid(x,y)



### LOS PROJECTION ---
    def projectLOS(self,alpha,theta,wrap=None):
        '''
            Project into LOS.
            Alpha is the azimuth angle in degrees counter clockwise bewteen East
             and the look direction from the target to the sensor.
            Theta is the incidence angle from vertical.
        '''
        self.LOS = True

        # Check inputs are valid
        if alpha is None:
            print('Specify --alpha (191 for asc; 169 for dsc)'); exit()
        if theta is None:
            print('Specify --theta (35)'); exit()

        # Convert to radians
        alpha = np.deg2rad(alpha)
        theta = np.deg2rad(theta)

        # Project into LOS
        self.Ulos = []
        for i in range(self.Nepochs):
            Ulos = self.Uup[i,:,:]*np.cos(theta)\
            +self.Ueast[i,:,:]*np.sin(theta)*np.cos(alpha)\
            +self.Unorth[i,:,:]*np.sin(theta)*np.sin(alpha)

            # Wrap if requested
            if wrap:
                Ulos %= wrap

            # Append to array
            self.Ulos.append(Ulos)
        self.Ulos = np.array(self.Ulos)




### DISPLACEMENT PLOTTING ---
    def plotDisplacements(self,cmap='RdBu_r',limits=None,palette=[None,None],
        steps=3,vscale=1):
        '''
            Plot displacements for all epochs.
            Data are provided as 3D arrays with the first dimension being the
             number of epochs (e.g., [Nepochs x 256 x 256])
        '''
        # Number of subplots
        m = n = int(np.ceil(np.sqrt(self.Nepochs)))

        # Build figure
        Fig = plt.figure()
        for i in range(self.Nepochs):
            # Create axis
            ax = Fig.add_subplot(m,n,i+1)

            # Plot vertical data
            cax = ax.imshow(self.Uup[i,:,:],
                cmap=cmap,vmin=palette[0],vmax=palette[1],
                extent=self.extent)

            # Plot horizontal vectors
            U = self.Ueast[i,::steps,::steps]
            V = self.Unorth[i,::steps,::steps]
            ax.quiver(self.X[::steps,::steps],self.Y[::steps,::steps],U,V,color='k',
                units='xy',scale=vscale,headwidth=3,headlength=5)

            # Format axis
            if limits:
                ax.set_xlim(limits[:2])
                ax.set_ylim(limits[2:])
            ax.set_aspect(1)
            Fig.colorbar(cax,orientation='vertical')
        Fig.tight_layout()
        Fig.savefig(os.path.join(self.name,'Displacements.pdf'),type='pdf')

    def plotLOS(self,cmap='RdBu_r',limits=None,palette=[None,None]):
        '''
            Plot displacements for all epochs.
            Data are provided as 3D arrays with the first dimension being the
             number of epochs (e.g., [Nepochs x 256 x 256])
        '''
        # Number of subplots
        m = n = int(np.ceil(np.sqrt(self.Nepochs)))

        # Build figure
        Fig = plt.figure(figsize=(20,10))
        for i in range(self.Nepochs):
            # Create axis
            ax = Fig.add_subplot(m,n,i+1)

            # Plot LOS displacement data
            cax = ax.imshow(self.Ulos[i,:,:],
                cmap=cmap,vmin=palette[0],vmax=palette[1],
                extent=self.extent)

            # Format axis
            if limits:
                ax.set_xlim(limits[:2])
                ax.set_ylim(limits[2:])
            ax.set_aspect(1)
            Fig.colorbar(cax,orientation='vertical')
        Fig.tight_layout()
        Fig.savefig(os.path.join(self.name,'LOSdisplacements.pdf'),type='pdf')



### MAIN FUNCTION ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()

    # Load displacement maps from Relax data
    eastFiles, northFiles, upFiles = detectFiles(inps.fldr,dspType=inps.dspType,
        verbose=inps.verbose)

    # Format data
    DSname = os.path.basename(inps.fldr)
    DS = displacementDataset(DSname,eastFiles,northFiles,upFiles,
        dataType=inps.dspType,verbose=inps.verbose)

    # Plot data
    DS.plotDisplacements(cmap=inps.cmap,limits=inps.limits,palette=inps.palette,
        steps=inps.steps,vscale=inps.vscale)

    # Project into LOS if requested
    if inps.LOS == True:
        # Project into LOS
        DS.projectLOS(inps.alpha,inps.theta,wrap=inps.wrap)

        # Plot LOS
        DS.plotLOS(cmap=inps.cmap,limits=inps.limits,palette=inps.palette)


    plt.show()