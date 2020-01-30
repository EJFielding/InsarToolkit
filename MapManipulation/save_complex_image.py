#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

'''
	Save complex image to 2-band amplitude and phase
'''



def createParser():
    import argparse
    parser = argparse.ArgumentParser(description='Save complex image to 2-band amplitude and phase')
    parser.add_argument(dest='inputImg',type=str,help='Input image')
    parser.add_argument(dest='outputImg',type=str,help='Output image')

    parser.add_argument('--nodata',dest='nodata',type=float,help='Output nodata value.')
    parser.add_argument('-p','--plot',dest='plot',action='store_true',help='Plot before saving')
    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


if __name__=='__main__':
	## Script inputs 
	inpt=cmdParser()

	## Read in input image
	DSin=gdal.Open(inpt.inputImg,gdal.GA_ReadOnly)
	Img=DSin.GetRasterBand(1).ReadAsArray()
	Proj=DSin.GetProjection()
	T=DSin.GetGeoTransform()
	M=DSin.RasterYSize; N=DSin.RasterXSize
	left=T[0]; dx=T[1]; right=left+dx*N 
	top=T[3]; dy=T[5]; bottom=top+dy*M 
	extent=(left, right, bottom, top)

	# Convert to amplitude and phase
	Amp=np.abs(Img)
	Phs=np.angle(Img)

	## Save image values
	Driver=gdal.GetDriverByName('GTiff')

	OutDS=Driver.Create(inpt.outputImg,N,M,2,gdal.GDT_Float32) 
	OutDS.GetRasterBand(1).WriteArray(Amp)
	OutDS.GetRasterBand(2).WriteArray(Phs) 
	OutDS.SetProjection(Proj) 
	OutDS.SetGeoTransform(T) 
	OutDS.FlushCache() 


	## Plot if requested
	if inpt.plot is True:
		from ImageTools import imgStats
		AmpStats=imgStats(Amp**0.5,pctmin=2,pctmax=98)

		F=plt.figure()
		axAmp=F.add_subplot(121)
		caxAmp=axAmp.imshow(Amp**0.5,cmap='Greys',vmin=AmpStats.vmin,vmax=AmpStats.vmax,extent=extent)
		F.colorbar(caxAmp,orientation='horizontal')
		axPhs=F.add_subplot(122)
		caxPhs=axPhs.imshow(Phs,cmap='jet',extent=extent)
		F.colorbar(caxPhs,orientation='horizontal')

		plt.show()