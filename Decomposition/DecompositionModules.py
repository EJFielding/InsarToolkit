'''
These are broken into subroutines:
* Load data
* Resample data into same area
* Convert azimuth, incidence angle to 3D pointing vectors
* Solve for EW and vert signals
'''

### IMPORT MODULES ---
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from osgeo import gdal
from LookAngleVectors import ISCE_look_to_vector



### GENERIC ---
def formulateOutnames(Fmt,Datasets):
    '''
    Formulate output names.
    '''
    fldr = Fmt['outFolder']

    # Empty list
    outputNames = []
    for dataset in Datasets:
        fname = os.path.basename(dataset['name'])
        fname = 'Resampled_{:s}'.format(fname)
        outputNames.append(os.path.join(fldr,fname))

    return outputNames


def GDALextent(DS):
    '''
    Get the resampling bounds and imshow extent.
     bounds = (xmin ymin xmax ymax)
     extent = (left right bottom top)
    '''
    [xmin,dx,xshear,ymax,yshear,dy] = DS.GetGeoTransform()
    M,N = DS.RasterYSize, DS.RasterXSize

    xmax = xmin+dx*N
    ymin = ymax+dy*M

    bounds = [xmin, ymin, xmax, ymax]
    extent = [xmin, xmax, ymin, ymax]

    return bounds, extent


def GDALcommon_bounds(boundsList):
    '''
    Find the conservative common bounds in gdalwarp te format 
     (xmin,ymin,xmax,ymax).
    '''
    # Reformat bounds list into 2D array
    boundsList = np.array(boundsList)
    print('Data set bounds')
    [print(bounds) for bounds in boundsList]

    # Find conservative bounds
    xmin = boundsList[:,0].max()
    ymin = boundsList[:,1].max()
    xmax = boundsList[:,2].min()
    ymax = boundsList[:,3].min()

    globalBounds = (xmin, ymin, xmax, ymax)

    return globalBounds


def GDALresample(names,dataList,bounds):
    '''
    Resample based on given geographic bounds.
    INPUTS
        names is a list of output filenames
        dataList is a list of gdal data sets within the common bounds
        bounds are the common bounds to which to warp the maps
    OUTPUTS
        ResampledDatasets is a list of gdal data sets resampled to a common
         extent
    '''
    resampledData = []

    for n,name in enumerate(names):
        print('Resampling {:s}'.format(name))
        
        resampledData.append(gdal.Warp(name,dataList[n],
            options=gdal.WarpOptions(format='VRT',outputBounds=bounds)))

    return resampledData


## Decomposition
def decompose(Lx,Ly,Lz,Ulos):
    '''
    Decompose (recompose) a data set into EW, (NS,) and vertical components by
     inverting a design matrix. Leverage as many data sets as available.
    Option for 3D decomposition coming soon... Caution is warranted.

    SETUP
     | pAx  pAz |.| ux | = | rA |
     | pDx  pDz | | uz |   | rD |
           P     .   u   =    r
    
       =>  u = Pinv.r

    INPUTS
        * This function works for D data sets, each covering a common M x N 
         area
        Lx is a DxMxN array for the x-component of pointing vectors
        Ly is a DxMxN array for the y-component of pointing vectors
        Lz is a DxMxN array for the z-component of pointing vectors
        Ulos is a DxMxN array for the LOS displacements
    '''
    # Sanity check and number of data sets
    if Lx.shape == Ly.shape == Lz.shape == Ulos.shape:
        print('Data shape: {:d} x {:d} x {:d}'.format(*Lx.shape))

        nDatasets, M, N = Lx.shape
        MN = M*N
        print('Continuing with {:d} data sets'.format(nDatasets))

    # Decompose signals
    print('Inverting for pixels...')

    # Iterate on pixel-by-pixel basis
    U = np.zeros((2,M,N))
    for i,j in itertools.product(range(M),range(N)):
        # Design matrix (pointing vectors)
        P = np.array([Lx[:,i,j],Lz[:,i,j]]).T
        Pinv = np.linalg.inv(np.dot(P.T,P)).dot(P.T)

        # LOS observations
        r = Ulos[:,i,j]

        # Invert for displacements
        u = Pinv.dot(r)
        U[0,i,j] = u[0]
        U[1,i,j] = u[1]

    return U



### ISCE ---
def decomposeISCE(Fmt,Datasets,plot=False):
    '''
    Decompose a typical georeferenced ISCE data set.
    INPUTS
        Fmt
        Datasets
    OUTPUTS
    '''
    print('ISEC, ISCE, baby')

    nDatasets = len(Datasets)

    ## Formulate filenames for output files
    outNames = formulateOutnames(Fmt,Datasets)
    IFGnames = [outName+'_IFG.vrt' for outName in outNames]
    LOSnames = [outName+'_LOS.vrt' for outName in outNames]


    ## Load data
    IFGdatasets, LOSdatasets, boundsList = ISCE_loadDatasets(Datasets)
    print('*'*31)


    ## Resample spatial extent
    # Determine common bounds
    globalBounds = GDALcommon_bounds(boundsList)
    print('Global bounds: {}'.format(globalBounds))

    # Resample to bounds
    print('Resampling IFGs:')
    IFGdatasets = GDALresample(IFGnames,IFGdatasets,globalBounds)

    print('Resampling LOSs:')
    LOSdatasets = GDALresample(LOSnames,LOSdatasets,globalBounds)
    print('*'*31)


    ## Plot if requested
    if plot == True:
        # Plot resampled inputs
        print('Preparing plots')
        for n in range(nDatasets):
            ISCE_plotIFG(IFGdatasets[n],LOSdatasets[n])
        print('*'*31)


    ## Convert to 3D look vectors
    print('Converting LOS angles to 3D vectors')
    Lx = []; Ly = []; Lz = []
    for n in range(nDatasets):
        lx, ly, lz = ISCE_look_to_vector(LOSdatasets[n])
        Lx.append(lx)
        Ly.append(ly)
        Lz.append(lz)
    Lx = np.array(Lx)
    Ly = np.array(Ly)
    Lz = np.array(Lz)
    print('*'*31)


    ## Decompose into EW, NS, and vert components
    # Create a 3D array containing the LOS displacement data sets
    Ulos = np.array([dataset.GetRasterBand(2).ReadAsArray() for dataset in \
        IFGdatasets])

    # Decompose into motion components
    print('Decomposing')
    U = decompose(Lx,Ly,Lz,Ulos)

    # Save results
    saveName_matrix = '{:s}_DisplacementMatrix'.format(os.path.join(Fmt['outFolder'],
        Fmt['projectName']))
    print('Decomposition finished. Saving to {:s}'.format(saveName_matrix))
    np.savez(saveName_matrix,U=U)
    print('*'*31)

    # Plot final results
    ISCE_plotResults(IFGdatasets,U)


def ISCE_loadDatasets(Datasets):
    '''
    Load georeferenced data sets in the ISCE format.
    This function draws on ISCE_loadIFG and ISCE_loadLOS.
    '''

    # Empty lists
    IFGdatasets = []
    LOSdatasets = []
    boundsList = []

    # Load data sets
    for dataset in Datasets:
        # Load amplitude and phase
        IFGdatasets.append(ISCE_loadIFG(dataset))

        # Load incidence and azimuth
        LOSdatasets.append(ISCE_loadLOS(dataset))

        # Record extent
        bounds,extent = GDALextent(IFGdatasets[-1])
        boundsList.append(bounds)

    return IFGdatasets, LOSdatasets, boundsList


def ISCE_loadIFG(dataset):
    '''
    Load the two-band ISCE IFG file.
    '''
    # File name
    fldr = dataset['folder']
    ifgName = os.path.join(fldr,'filt_topophase_2stage.unw.geo.vrt')

    print('Loading {:s}\n... IFG: {:s}'.format(dataset['name'],ifgName))

    # Load GDAL data set
    DSifg = gdal.Open(ifgName,gdal.GA_ReadOnly)

    return DSifg


def ISCE_loadLOS(dataset):
    '''
    Load the two-band ISCE LOS file.
    '''
    # File name
    fldr = dataset['folder']
    losName = os.path.join(fldr,'los.rdr.geo.vrt')

    print('... LOS: {:s}'.format(losName))

    # Load GDAL data set
    DSlos = gdal.Open(losName,gdal.GA_ReadOnly)

    return DSlos


def ISCE_plotIFG(DSifg,DSlos):
    '''
    Plot the amplitude and phase of the IFG.
    '''
    # Parse amplitude and phase maps
    amp = DSifg.GetRasterBand(1).ReadAsArray()
    phs = DSifg.GetRasterBand(2).ReadAsArray()

    # Mask IFG data
    phs[np.isnan(phs)==1]=0.

    amp = np.ma.array(amp,mask=(amp==0.))
    phs = np.ma.array(phs,mask=(phs==0.))

    # Scale and clip by percent
    amp = amp**0.05
    ampMin,ampMax = np.percentile(amp.compressed().flatten(),[0,90])
    phsMin,phsMax = np.percentile(phs.compressed().flatten(),[0,100])

    # Establish plot
    Fig,[axIFG,axAz,axInc] = plt.subplots(ncols=3)

    # Plot IFG data
    cAmp = axIFG.imshow(amp**0.5,cmap='Greys_r',vmin=ampMin,vmax=ampMax)
    cPhs = axIFG.imshow(phs,cmap='jet',vmin=phsMin,vmax=phsMax,alpha=0.4)

    # Parse azimuth and incidence data
    inc = DSlos.GetRasterBand(1).ReadAsArray()
    az = DSlos.GetRasterBand(2).ReadAsArray()

    # Mask LOS data
    inc = np.ma.array(inc,mask=(inc==0.))
    az = np.ma.array(az,mask=(az==0.))

    # Clip by percent
    azMin,azMax = np.percentile(az.compressed().flatten(),[1,99])
    incMin,incMax = np.percentile(inc.compressed().flatten(),[1,99])

    # Plot LOS data
    cInc = axInc.imshow(inc,cmap='viridis',vmin=incMin,vmax=incMax)
    cAz = axAz.imshow(az,cmap='viridis',vmin=azMin,vmax=azMax)

    # Format plot
    axIFG.set_xticks([]); axIFG.set_yticks([])
    axIFG.set_aspect(1)
    axIFG.set_title('IFG')
    Fig.colorbar(cPhs,ax=axIFG,orientation='horizontal')

    axInc.set_xticks([]); axInc.set_yticks([])
    axInc.set_aspect(1)
    axInc.set_title('Inc')
    Fig.colorbar(cInc,ax=axInc,orientation='horizontal')

    axAz.set_xticks([]); axAz.set_yticks([])
    axAz.set_aspect(1)
    axAz.set_title('Az')
    Fig.colorbar(cAz,ax=axAz,orientation='horizontal')

def ISCE_plotResults(DSifg,U):
    '''
    INPUTS
        DSifg are the interferogram data sets
        U is the 2xMxN results of the decomposition
    '''
    # Create mask for both phase maps
    M = DSifg[0].RasterYSize
    N = DSifg[0].RasterXSize
    mask = np.ones((M,N))
    for dataset in DSifg:
        phs = dataset.GetRasterBand(2).ReadAsArray()
        mask[phs==0.]=0

    Fig, [axEW,axVt] = plt.subplots(ncols = 2, figsize=(11,6))

    # Plot EW component
    EW = np.ma.array(U[0,:,:],mask=(mask==0))

    cEW = axEW.imshow(EW,cmap='jet')
    axEW.set_aspect(1)
    Fig.colorbar(cEW,ax=axEW,orientation='horizontal')

    # Plot vertical component
    Vt = np.ma.array(U[1,:,:],mask=(mask==0))

    cVt = axVt.imshow(Vt,cmap='jet')
    axVt.set_aspect(1)
    Fig.colorbar(cVt,ax=axVt,orientation='horizontal')
