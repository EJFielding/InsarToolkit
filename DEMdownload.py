#!/usr/bin/env python3 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For downloading the most up-to-date SRTM DEM 
# Author: Rob Zinke, after ARIA-tools "extractProduct.py" 
#  by S Sangha and D Bekaert  
# For personal use only
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys
import numpy as np 
from osgeo import gdal 

_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')

def createParser():
    '''
        Extract specified product layers. The default is to export all layers.
    '''

    import argparse
    parser = argparse.ArgumentParser(description='Program to extract data and meta-data layers from ARIA standard GUNW products. Program will handle cropping/stiching when needed. By default, the program will crop all IFGs to bounds determined by the common intersection and bbox (if specified)')
    parser.add_argument('-w', '--workdir', dest='workdir', default='./', help='Specify directory to deposit all outputs. Default is local directory where script is launched.')
    parser.add_argument('-p', '--projection', dest='proj', default='WGS84', type=str,
            help='projection for DEM. By default WGS84.')
    parser.add_argument('-b', '--bounds', dest='bounds', type=str, default=None, help="Example : 'W S E N'")
    parser.add_argument('-of', '--outputFormat', dest='outputFormat', type=str, default='VRT', help='GDAL compatible output format (e.g., "ENVI", "GTiff"). By default files are generated virtually except for "bPerpendicular", "bParallel", "incidenceAngle", "lookAngle","azimuthAngle", "unwrappedPhase" as these are require either DEM intersection or corrections to be applied')
    parser.add_argument('-verbose', '--verbose', action='store_true', dest='verbose', help="Toggle verbose mode on.")

    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)

# Make DEM 
def prep_dem(demfilename, bounds, proj, arrshape=None, workdir='./', outputFormat='ENVI'):
    '''
        Function which load and export DEM, lat, lon arrays.
        If "Download" flag is specified, DEM will be donwloaded on the fly.
    '''

    # File must be physically extracted, cannot proceed with VRT format. Defaulting to ENVI format.
    if outputFormat=='VRT':
       outputFormat='ENVI'

    # Download DEM
    demfilename=os.path.join(workdir,'SRTM_3arcsec'+'.dem')
    gdal.Warp(demfilename, '/vsicurl/'+_world_dem, options=gdal.WarpOptions(format=outputFormat, outputBounds=bounds, outputType=gdal.GDT_Int16, xRes=0.000277778, yRes=0.000277778, dstNodata=-32768.0, srcNodata=-32768.0))
    gdal.Translate(demfilename+'.vrt', demfilename, options=gdal.TranslateOptions(format="VRT")) #Make VRT


if __name__ == '__main__':
    """
    Main driver
    """
    inps = cmdLineParse() 

    # Output directory 
    if not os.path.exists(inps.workdir): 
    	os.makedirs(inps.workdir)

    # Format bounds 
    bounds=inps.bounds.strip('(').strip(')') 
    bounds=bounds.split(',')
    bounds=(float(bounds[0]),float(bounds[1]),float(bounds[2]),float(bounds[3]))
    print('Directory: %s' % (inps.workdir)) 
    print('Projection: %s' % (inps.proj)) 
    print('Bounds',bounds) 
    print('Format: %s' % (inps.outputFormat)) 

    # Create DEM 
    demname='SRTM_DEM_{}_{}_{}_{}'.format(str(bounds[0]),str(inps.bounds[1]),str(inps.bounds[2]),str(inps.bounds[3])) 
    prep_dem(demfilename=demname,
    	bounds=bounds,
    	proj=inps.proj,
    	workdir=inps.workdir,
    	outputFormat=inps.outputFormat)

