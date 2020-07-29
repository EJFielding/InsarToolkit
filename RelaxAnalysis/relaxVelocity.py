#!/usr/bin/env python3
'''
Use outputs of relax2LOS to compute LOS velocity.
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

    # Required inputs
    inputArgs = parser.add_argument_group('ESSENTIAL ARGUMENTS')
    inputArgs.add_argument(dest='fname', type=str,
        help='List of cumulative LOS displacement files and times')
    inputArgs.add_argument('-o','--outName', dest='outName', type=str, default='aveVeloc',
        help='Output name')

    # Plot arguments
    plotArgs = parser.add_argument_group('PLOT ARGUMENTS')
    plotArgs.add_argument('-c','--cmap', dest='cmap', default='jet',
        help='Colormap')

    inputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        default='Verbose mode')
    return parser

def cmdParser(inpt_args=None):
    parser = createParser()
    return parser.parse_args(args=inpt_args)



### ANCILLARY FUNCTIONS ---
def loadData(dspList):
    '''
        Load data into
         Ucum = 3D array of cumulative displacements
         T = list of times in years
    '''
    # Empty lists
    Ucum = []
    T = []

    # Read displacement list
    with open(dspList,'r') as dspFile:
        lines = dspFile.readlines()
        dspFile.close()

    lines = [line.strip('\n') for line in lines if line[0] != '#']

    for line in lines:
        print(line)
        # Parse line
        line = line.split(' ')

        # Record time
        T.append(float(line[1]))

        # Open gdal data set
        DS = gdal.Open(line[0],gdal.GA_ReadOnly)

        Ucum.append(DS.GetRasterBand(1).ReadAsArray())

    # Spatial parameters
    M = DS.RasterYSize
    N = DS.RasterXSize
    tnsf = DS.GetGeoTransform()
    left=tnsf[0]; dx=tnsf[1]; right=left+dx*N
    top=tnsf[3]; dy=tnsf[5]; bottom=top+dy*M
    extent=(left, right, bottom, top)

    params = {}
    params['proj'] = DS.GetProjection()
    params['tnsf'] = tnsf
    params['M'] = M
    params['N'] = N
    params['extent'] = extent

    # Format at 3D array
    Ucum = np.array(Ucum)

    return Ucum, T, params


def aveVelocity(Ucum,T):
    '''
        Compute the average velocity.
    '''
    # Net displacement
    Unet = Ucum[-1,:,:] - Ucum[0,:,:]

    # Time difference
    Tnet = T[-1] - T[0]

    # Average velocity
    Vave = Unet/Tnet

    return Vave


def plotInputs(Ucum,T,params,m=4,n=5,cmap='jet'):
    '''
        Plot cumulative displacement time steps.
    '''
    Fig = plt.figure()
    for i in range(Ucum.shape[0]):
        ax = Fig.add_subplot(m,n,i+1)
        cax = ax.imshow(Ucum[i,:,:],cmap=cmap,
            vmin=-0.06,vmax=0.06)
        Fig.colorbar(cax,orientation='vertical')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('T = {}'.format(T[i]))

    Fig.suptitle('Cumulative displacement')


def plotAveVelocity(Vave,params,cmap='jet'):
    '''
        Plot average velocity computed by aveVelocity step.
    '''
    Fig = plt.figure()
    ax = Fig.add_subplot(111)

    cax = ax.imshow(Vave,cmap=cmap,
        extent=params['extent'])
    Fig.colorbar(cax,orientation='vertical')

    ax.set_title('Ave LOS velocity')


def saveVelocity(Vave,params,outName):
    '''
        Save LOS velocity to GeoTIFF.
    '''
    if outName[-4:] != '.tif': outName+='.tif'

    driver=gdal.GetDriverByName('GTiff')
    DSout=driver.Create(outName,params['N'],params['M'],1,
        gdal.GDT_Float32)
    DSout.GetRasterBand(1).WriteArray(Vave)
    DSout.GetRasterBand(1).SetNoDataValue(0)
    DSout.SetProjection(params['proj'])
    DSout.SetGeoTransform(params['tnsf'])
    DSout.FlushCache()

    print('Saved to: {}'.format(outName))


### MAIN FUNCTION ---
if __name__ == '__main__':
    # Gather inputs
    inps = cmdParser()

    # Load data
    Ucum,T,params = loadData(inps.fname)

    # Calculate average velocity
    Vave = aveVelocity(Ucum,T)

    # Plot inputs
    plotInputs(Ucum,T,params,m=4,n=5,cmap=inps.cmap)

    # Plot results
    plotAveVelocity(Vave,params,cmap=inps.cmap)

    # Save to GeoTIFF
    saveVelocity(Vave,params,inps.outName)

    plt.show()