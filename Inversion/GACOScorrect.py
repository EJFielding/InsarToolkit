#!/usr/bin/env python3
'''
Remove the tropospheric delay from a single ISCE interferogram using GACOS data.
'''


### IMPORT MODULES ---
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


### PARSER ---
def createParser():
    Description = '''Plot GDAL-compatible map data sets, including complex
images and multi-band data sets.'''

    parser=argparse.ArgumentParser(description = Description,
        formatter_class = argparse.RawTextHelpFormatter)

    # Input arguments
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    inputArgs.add_argument('-i','--ifg-name', dest='ifgName', type=str, required=True,
        help='IFG file')
    inputArgs.add_argument('-l','--los-name', dest='losName', type=str, required=True,
        help='LOS file')
    inputArgs.add_argument('-r','--ref-model', dest='refName', type=str, required=True,
        help='Reference weather model')
    inputArgs.add_argument('-s','--sec-model', dest='secName', type=str, required=True,
        help='Secondary weather model')

    # Output arugments
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    outputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode')
    outputArgs.add_argument('-o','--outName', dest='outName', type=str,
        help='Output name')
    outputArgs.add_argument('--plot-weather-models', dest='plotModels', action='store_true',
        help='Verbose mode')

    return parser

def cmdParser(iargs = None):
    parser = createParser()

    return parser.parse_args(args = iargs)



### LOADING FUNCTIONS ---
def loadIFG(ifgName, verbose=False):
    '''
    Load interferogram from which the differential atmospheric signal is to be
     removed.
    INPUTS
        ifgName is the name of the IFG data set
    OUTPUTS
        IFGds is the GDAL data set
        ifg is the map of phase values, output directly because the band number
         depends on the data type
        extent is an object with spatial extent properties for plotting and
         geographic transformation
    '''
    if verbose == True:
        print('*'*32)
        print('Loading interferogram: {:s}'.format(ifgName))

    # Load GDAL data set
    IFGds = gdal.Open(ifgName, gdal.GA_ReadOnly)

    # Check that dataset is valid
    if IFGds is None:
        print('IFG data set not valid. Check filepath.')
        exit()

    # Load phase based on driver
    driver = IFGds.GetDriver().ShortName
    if verbose == True: print('Detected driver: {:s}'.format(driver))

    if driver == 'ISCE':
        phsBand = 2
    else:
        print('Driver {:s} not supported'.format(driver))
        exit()

    ifg = IFGds.GetRasterBand(phsBand).ReadAsArray()

    # Formulate mask
    mask = np.ones(ifg.shape)
    mask[np.isnan(ifg) == True] = 0

    # Spatial extent
    extent = GDALextent(IFGds, verbose=verbose)

    return IFGds, ifg, mask, extent

def loadLOS(LOSname, verbose=False):
    '''
    Load the line of sight (LOS) info corresponding to the IFG data set.
    The LOS data set must be registered in the same spatial reference frame as
     the IFG.
    Data format is detected automatically, as for the IFG.
    INPUTS
        LOS name is the name of the LOS file
    OUTPUTS
        inc is the map of incidence angle values
    '''
    if verbose == True:
        print('*'*32)
        print('Loading incidence angles: {:s}'.format(LOSname))

    # Extract incidence values
    LOSds = gdal.Open(LOSname, gdal.GA_ReadOnly)

    # Check that dataset is valid
    if LOSds is None:
        print('LOS data set not valid. Check filepath.')
        exit()

    # Load incidence based on driver
    driver = IFGds.GetDriver().ShortName
    if verbose == True: print('Detected driver: {:s}'.format(driver))

    if driver == 'ISCE':
        incBand = 1
    else:
        print('Driver {:s} not supported')
        exit()

    inc = LOSds.GetRasterBand(incBand).ReadAsArray()

    # Report stats if requested
    if verbose == True:
        I = np.ma.array(inc, mask=(inc == 0)).compressed()
        print('Mean: {:.4f}\nMin: {:.4f}\nMax: {:.4f}'. \
            format(I.mean(), I.min(), I.max()))

    return inc


def loadModel(GACOSname, extent, verbose=False, plot=False):
    '''
    Load GACOS weather model data and resample if necessary.
    First, load a data set using the loadGACOS function.
    Then, check that the extent equals that specified. Note, the extents are 
     specified as GDALextent objects.
    Resample to the specified extent and resolution if necessary using 
     GDALresample.
    '''
    if verbose == True:
        print('*'*32)
        print('Loading weather model data')

    # Load GACOS data set
    GACOSds = loadGACOS(GACOSname, verbose=verbose)
    GACOSextent = GDALextent(GACOSds)

    # Check that spatial extent of GACOS model matches that specified
    if GACOSextent.extent == extent.extent:
        if verbose == True:
            print('Data set extents equal. No resampling required.')
    else:
        if verbose == True:
            print('Data set extents not equal. Resampling required...')
        # Format output name
        outName = '{:s}_resampled'.format(GACOSname.strip('.ztd'))

        # Resample the GACOS data set
        GACOSds = GDALresample(GACOSds, extent, outName, 
            verbose=verbose, plot=plot)

    return GACOSds

def loadGACOS(gacosName,verbose=False):
    '''
    Load GACOS data set using GDAL.
    The GACOS RSC format is not readable by GDAL, so an ENVI .hdr file must be
     built from the .rsc. This is done automatically by the rsc2hdr function.
    INPUTS
        gacosName is the name of the .ztd file
    OUTPUTS
        GACOSds is the GDAL data set containing the GACOS model data
    '''
    if verbose == True: print('Loading GACOS model: {:s}'.format(gacosName))

    # Check if .xml file exists
    rscName = '{:s}.rsc'.format(gacosName)
    hdrName = '{:s}.hdr'.format(gacosName)
    if os.path.exists(hdrName):
        if verbose == True: print('... hdr exists')
    else:
        if verbose == True: print('... creating hdr')
        rsc2hdr(rscName, hdrName, verbose=verbose)

    # Load GDAL data set and confirm validity
    GACOSds = gdal.Open(gacosName, gdal.GA_ReadOnly)
    if GACOSds is None:
        print('GACOS data set not valid. Check filepath.')
        exit()

    return GACOSds

def rsc2hdr(rscName,hdrName,verbose=False):
    '''
    Build an ENVI .hdr file from a GACOS .rsc.
    '''
    # Read and format RSC contents
    with open(rscName) as inFile:
        text = inFile.readlines()
    lines = [line.split() for line in text]

    # Convert to dictionary for HDR formatting
    headers = dict(lines)

    # Add the filename such it can be called when making envi header
    headers['FILENAME'] = rscName.strip('.rsc')

    # Take the abs of the y-spacing as upper left corner is to be specified
    headers['Y_STEP'] = str(np.abs(float(headers['Y_STEP'])))

    # ENVI HDR content string
    enviHDR = '''ENVI
description = {{GACOS: {FILENAME} }}
samples = {WIDTH}
lines = {FILE_LENGTH}
bands = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
sensor type = Unknown
byte order = 0
map info = {{Geographic Lat/Lon, 1, 1, {X_FIRST}, {Y_FIRST}, {X_STEP}, {Y_STEP}, WGS-84, units=Degrees}}
coordinate system string = {{GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.017453292519943295]]}}'''.format(**headers)

    # Open HDR file and write outputs
    with open(hdrName, 'w') as enviHDRfile:
        enviHDRfile.write(enviHDR)

    if verbose == True: print('Built {:s} from {:s}'.format(hdrName, rscName))



### SPATIAL EXTENT AND RESAMPLING ---
class GDALextent:
    '''
    Parse the geospatial info from a GDAL data set for convenience.
    '''
    def __init__(self, DS, verbose=False):
        '''
        Get the extent of a GDAL data set in (left right bottom top) format.
        '''
        # Parse data set
        self.M, self.N = DS.RasterYSize, DS.RasterXSize
        self.left, self.dx, self.xshear, self.top, self.yshear, self.dy = \
            DS.GetGeoTransform()

        # Compute extent
        T = np.array([[ self.dx,     self.yshear],
                        [self.xshear, self.dy]])
        xy = np.array([[self.N, self.M]]).T
        xy0 = np.array([[self.left, self.top]]).T
        self.right, self.bottom = (T.dot(xy) + xy0).flatten()

        # Pyplot extent
        self.extent = (self.left, self.right, self.bottom, self.top)

        # GDAL warp extent
        self.te = (self.left, self.bottom, self.right, self.top)
        self.tr = (self.dx, self.dy)

        # Report if requested
        if verbose == True:
            extentStr = '''Geographic extent
    xmin: {:.4f}; xmax: {:.4f}; ymin: {:.4f}; ymax: {:.4f}'''
            print(extentStr.format(*self.extent))

def GDALresample(DS, extent, outName, verbose = True, plot=False):
    '''
    Resample the warp data set to the same spatial bounds and resolution as the
     example data set.
    INPUTS
        DS is the data set to be resampled
        extent is the GDALextent object of the reference extent
        outName is the name of the output data set
        plot if True will plot the WarpDS at its original and final resolution
    OUTPUTS
        WarpedDS is the WarpDS at it's final extent and resolution
    '''
    if verbose == True:
        resampCriteria = '''Resampling to
    shape: {M:d} x {N:d}
    x bounds: {left:.4f} - {right:.4f}
    y bounds: {bottom:.4f} - {top:.4f}
    x res, yres = {dx:.8f} x {dy:.8f}'''
        print(resampCriteria.format(**extent.__dict__))

    # Resample
    DSwarped = gdal.Warp(outName,DS,options=gdal.WarpOptions(format='envi',
        outputBounds=extent.te, xRes=extent.tr[0], yRes = extent.tr[1],
        resampleAlg='bilinear'))

    # Report if requested
    if verbose == True: print('Warped data set, saved to: {:s}'.format(outName))

    # Plot if requested
    if plot == True:
        Fig, [axOG, axWarp] = plt.subplots(ncols=2)
        caxOG = axOG.imshow(DS.GetRasterBand(1).ReadAsArray(),
            extent=GDALextent(DS).extent)
        caxWarp = axWarp.imshow(DSwarped.GetRasterBand(1).ReadAsArray(),
            extent=GDALextent(DSwarped).extent)

        axOG.set_aspect(1)
        axOG.set_title('Orig')
        Fig.colorbar(caxOG, ax=axOG, orientation='horizontal')
        axWarp.set_aspect(1)
        axWarp.set_title('Warped')
        Fig.colorbar(caxWarp, ax=axWarp, orientation='horizontal')
        Fig.suptitle('Warping results {:s}'.format(os.path.basename(outName)))

    return DSwarped


### ATMOSPHERIC REMOVAL ---
def zenith2slant(wavelen, inc, delay, verbose=False):
    '''
    Convert zenith atmospheric delay to slant-range delay.
    INPUTS
        wavelen is the radar wavelength in meters
        inc is the radar incidence angle
        delay is the GACOS delay in meters
    OUTPUTS
        slantDelay is the signal delay converted to slant range
    '''
    # Convert incidence angle to radians
    inc = np.deg2rad(inc)

    # Scaling factor
    scaling = -4*np.pi/wavelen

    # Project zenith delay into slant
    slantDelay = scaling/np.cos(inc)*delay

    return slantDelay

def saveIFG(IFGds, ifg_corr, outName, verbose=False):
    '''
    Save the corrected phase map to a data set of the same type with the given
     output name.
    '''
    # Save to file
    Driver = gdal.GetDriverByName(IFGds.GetDriver().ShortName)
    OutDS=Driver.Create(outName,
        IFGds.RasterXSize,IFGds.RasterYSize,
        2,gdal.GDT_Float64)
    OutDS.GetRasterBand(1).WriteArray(IFGds.GetRasterBand(1).ReadAsArray())
    OutDS.GetRasterBand(2).WriteArray(ifg_corr)
    OutDS.SetProjection(IFGds.GetProjection())
    OutDS.SetGeoTransform(IFGds.GetGeoTransform())
    OutDS.SetMetadata(IFGds.GetMetadata())
    OutDS.FlushCache()

    plt.figure()
    plt.imshow(OutDS.GetRasterBand(2).ReadAsArray())
    plt.show()
    exit()

    # Report if requested
    if verbose == True: print('Saved corrected IFG to {:s}'.format(outName))



### PLOTTING ---
def plotInputs(extent, mask, ifg, inc, refImg, secImg):
    '''
    Plot input data sets.
    All are assumed to be coregistered to the same spatial extent.
    INPUTS
        extent is the Pyplot extent (left, right, bottom, top)
        ifg = map of phase values
        inc = map of incidence angle values
        refImg = map of reference atmo delays
        secImg = map of secondary atmo delays
    '''
    # Spawn figure
    IptFig, [axIFG,axInc,axRef,axSec] = plt.subplots(figsize=(12,6), ncols=4)

    # Plot original interferogram
    caxIFG = axIFG.imshow(np.ma.array(ifg, mask=(mask==0)),
        cmap='jet', extent=extent)

    axIFG.set_aspect(1)
    axIFG.set_title('Phase (incl. atmo)')
    cbarIFG = IptFig.colorbar(caxIFG, ax=axIFG, orientation='horizontal')
    cbarIFG.set_label('(radians)')

    # Plot incidence angles
    caxInc = axInc.imshow(np.ma.array(inc, mask=(mask==0)),
        cmap='viridis', extent=extent)

    axInc.set_aspect(1)
    axInc.set_title('Incidence angle')
    cbarInc = IptFig.colorbar(caxInc, ax=axInc, orientation='horizontal')
    cbarInc.set_label('(degrees)')

    # Plot reference atmo delay
    AtmoMin = np.min([refImg.min(), secImg.min()])
    AtmoMax = np.max([refImg.max(), secImg.max()])

    caxRef = axRef.imshow(refImg, vmin=AtmoMin, vmax=AtmoMax,
        cmap='cividis', extent=extent)

    axRef.set_aspect(1)
    axRef.set_title('Ref delay')
    cbarRef = IptFig.colorbar(caxRef, ax=axRef, orientation='horizontal')
    cbarRef.set_label('(m)')

    # Plot secondary atmo delay
    caxSec = axSec.imshow(secImg, vmin=AtmoMin, vmax=AtmoMax,
        cmap='cividis', extent=extent)

    axSec.set_aspect(1)
    axSec.set_title('Sec delay')
    cbarSec = IptFig.colorbar(caxSec, ax=axSec, orientation='horizontal')
    cbarSec.set_label('(m)')

    # Format plot
    IptFig.suptitle('Input data')
    IptFig.tight_layout()

def plotOutputs(extent, mask, OGifg, delay, Cifg):
    '''
    Plot the results of the GACOS tropospheric correction.
    '''
    # Spawn figure
    OutFig, [axOG, axAtmo, axCorr] = plt.subplots(figsize=(10,6), ncols=3)

    # Plot original interferogram
    OGifg = np.ma.array(OGifg, mask=(mask==0))
    IFGmin, IFGmax = np.percentile(OGifg.compressed().flatten(), (1,99))
    caxOG = axOG.imshow(OGifg, vmin=IFGmin, vmax=IFGmax,
        cmap='jet', extent=extent)

    axOG.set_aspect(1)
    axOG.set_title('Orig. ifg')
    cbarOG = OutFig.colorbar(caxOG, ax=axOG, orientation='horizontal')
    cbarOG.set_label('(m)')

    # Plot tropospheric delay
    delay = np.ma.array(delay, mask=(mask==0))
    AtmoMax = np.max(np.abs(delay.compressed().flatten())); AtmoMin = -AtmoMax
    caxAtmo = axAtmo.imshow(delay, vmin=AtmoMin, vmax=AtmoMax,
        cmap='jet', extent=extent)

    axAtmo.set_aspect(1)
    axAtmo.set_title('Tropo delay')
    cbarAtmo = OutFig.colorbar(caxAtmo, ax=axAtmo, orientation='horizontal')
    cbarAtmo.set_label('(m)')

    # Plot corrected interferogram
    Cifg = np.ma.array(Cifg, mask=(mask==0))
    IFGmin, IFGmax = np.percentile(Cifg.compressed().flatten(), (1,99))
    caxCorr = axCorr.imshow(Cifg, vmin=IFGmin, vmax=IFGmax,
        cmap='jet', extent=extent)

    axCorr.set_aspect(1)
    axCorr.set_title('Corr. ifg')
    cbarCorr = OutFig.colorbar(caxCorr, ax=axCorr, orientation='horizontal')
    cbarCorr.set_label('(m)')

    # Format plot
    OutFig.suptitle('Correction')
    OutFig.tight_layout()



### MAIN ---
if __name__=='__main__':
    # Gather arguments
    inps = cmdParser()

    # Load ifg
    IFGds, ifg, mask, extent = loadIFG(inps.ifgName, verbose=inps.verbose)

    # Load incidence angle
    inc = loadLOS(inps.losName, verbose=inps.verbose)

    # Load weather models and resample if necessary
    REFds = loadModel(inps.refName, extent, verbose=inps.verbose,
        plot=inps.plotModels)
    SECds = loadModel(inps.secName, extent, verbose=inps.verbose,
        plot=inps.plotModels)

    # Plot inputs
    REFdelay = REFds.GetRasterBand(1).ReadAsArray()
    SECdelay = SECds.GetRasterBand(1).ReadAsArray()

    plotInputs(extent.extent, mask, ifg, inc, REFdelay, SECdelay)

    # Project delay into LOS and radians
    wavelen = 0.0556  # radar wavelength in m, hard code for now
    REFdelay = zenith2slant(wavelen, inc, REFdelay, verbose=inps.verbose)
    SECdelay = zenith2slant(wavelen, inc, SECdelay, verbose=inps.verbose)

    # Compute differential delay
    DiffDelay = REFdelay - SECdelay

    # Save differential delay
    saveIFG(IFGds, DiffDelay, './atmoscreen')

    # Apply correction
    ifg_corr = ifg - DiffDelay
    ifg_corr[mask==0] = np.nan

    # Save corrected IFG to file
    saveIFG(IFGds, ifg_corr, inps.outName, verbose=inps.verbose)

    # Plot outputs
    plotOutputs(extent.extent, mask, ifg, DiffDelay, ifg_corr)


    plt.show()