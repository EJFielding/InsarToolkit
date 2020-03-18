#!/usr/bin/env python3
"""
	Compute the expected velocity coefficient based on the
	 satellite LOS and the local topography.
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from osgeo import gdal
from geoFormatting import GDALtransform
from slopeFunctions import satGeom2vect3d, computeGradients, makePointingVectors
from viewingFunctions import mapPlot


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Given a rasterized DEM in Cartesian coordinates (e.g., UTM), compute the coefficient of expected velocity, i.e., LOS sensitivity times slope factor.')

	# Input data
	parser.add_argument('-d','--dem', dest='DEMname', type=str, required=True, help='Name of DEM in Cartesian coordinates.')
	parser.add_argument('-losAz','--los-azimuth', dest='losAz', type=float, required=True, help='Line of sight azimuth following ARIA convention.')
	parser.add_argument('-losInc','--los-incidence', dest='losInc', type=float, required=True, help='Line of sight incidence angle (from vertical) following ARIA convention.')

	# Outputs
	parser.add_argument('-o','--outName', dest='outName', type=str, required=True, help='Output name base')
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot outputs')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## Plot LOS vector in 3D
def plotLOSvector(Lx,Ly,Lz):
	Fig=plt.figure()
	ax=Fig.add_subplot(111,projection='3d')
	ax.quiver(0,0,0,Lx,Ly,Lz,color='k')
	ax.plot([-1,1],[0,0],[0,0],'k')
	ax.plot([0,0],[-1,1],[0,0],'k')
	ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1]); ax.set_zlim([-1.1,1.1])
	ax.set_xlabel('--easting--'); ax.set_ylabel('--northing--')


## Save georeferenced map
def saveMap(templateDS,band,savename):
	"""
		Provide a template data set with same spatial extent and 
		 resolution as images to be saved.
	"""

	# Gather parameters from template dataset
	N=templateDS.RasterXSize; M=templateDS.RasterYSize
	Proj=templateDS.GetProjection()
	Tnsf=templateDS.GetGeoTransform()

	# Save image to geotiff
	driver=gdal.GetDriverByName('GTiff')
	DSout=driver.Create(savename,N,M,1,gdal.GDT_Float32)
	DSout.GetRasterBand(1).WriteArray(band)
	DSout.SetProjection(Proj)
	DSout.SetGeoTransform(Tnsf)
	DSout.FlushCache()



### MAIN CALL ---
if __name__=='__main__':
	inpt=cmdParser()

	## Load DEM
	elevDS=gdal.Open(inpt.DEMname,gdal.GA_ReadOnly)
	elev=elevDS.GetRasterBand(1).ReadAsArray()
	tnsf=elevDS.GetGeoTransform(); T=GDALtransform(elevDS)

	if inpt.verbose is True: 
		print('Loaded DEM: {}'.format(inpt.DEMname))
		print('DEM size: {}'.format(elev.shape))

	if inpt.plot is True:
		mapPlot(elev,background='auto',cmap='terrain',cbar_orientation='horizontal',
			extent=T.extent,showExtent=True,title='Elevation')


	## Convert look angles into LOS vector L
	# azimuth measured in degrees CCW from east
	# incidence angle measured from vertical
	Lx,Ly,Lz=satGeom2vect3d(inpt.losAz,inpt.losInc)

	# Plot LOS pointing vector if requested
	if inpt.plot is True:
		plotLOSvector(Lx,Ly,Lz)

	# Flip direction of LOS vector to point from target to sensor
	#  This way, toward the satellite is positive
	Lx=-Lx; Ly=-Ly; Lz=-Lz



	### PROJECTION ---
	"""
		First, compute the vector pointing directly downhill at each pixel
		 of the input DEM. 
		Then, determine the expected velocity in satellite LOS if the
		 ground were moving directly along that vector (sensitivity coefficient)
		Finally, scale the expected LOS velocity by the slope at each pixel
	"""

	## Compute downhill vectors
	# Compute dzdx and dzdy at each pixel
	gradients=computeGradients(elev,dx=T.xstep,dy=T.ystep)

	# Compute vectors pointing downhill at each pixel
	Dx,Dy,Dz=makePointingVectors(gradients)


	## Sensitivity coefficient
	# Find the dot product of LOS vector and downhill vectors
	#  L.D = LxDx + LyDy + LzDz = W(mxn)
	W=Lx*Dx+Ly*Dy+Lz*Dz


	# Save sensitivity coefficient to data set
	W=np.nan_to_num(W)
	sensitivityName='{}_sensitivityCoeff.tif'.format(inpt.outName)
	saveMap(elevDS,band=W,savename=sensitivityName)

	# Plot sensitivity coefficient
	if inpt.plot is True:
		mapPlot(W,background='auto',cmap='coolwarm',cbar_orientation='horizontal',
				extent=T.extent,showExtent=True,title='Sensitivity (LOS.Downhill)')


	## Scale by vertical component
	# If ground velocity is gravity-driven, it should scale with vertical component
	Dz=np.nan_to_num(Dz)
	VX=W*-Dz # expected velocity


	# Save expected velocity to data set
	sensitivityName='{}_expectedVelocityCoeff.tif'.format(inpt.outName)
	saveMap(elevDS,band=VX,savename=sensitivityName)

	# Plot expected velocity
	if inpt.plot is True:
		mapPlot(-Dz,background='auto',cmap='viridis',cbar_orientation='horizontal',
				extent=T.extent,showExtent=True,title='Slope (-Dz)')		

		mapPlot(VX,background='auto',cmap='Spectral_r',cbar_orientation='horizontal',
				extent=T.extent,showExtent=True,
				title='Expected velocity coefficient: (Sensitivity x Slope)')


	plt.show()