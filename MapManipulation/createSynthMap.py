#!/usr/bin/env python3
"""
	Make synthetic georeferenced data sets
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal


### INPUT CLASS ---
class inputs:
	def __init__(self): 
		# Map options
		self.basevalue=1 # base value of blank map

		# Geography options
		self.Projection='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
		self.LatStart=34.0 # c. Los Angeles
		self.LonStart=118.0
		self.dx=0.000833 # 3 arc sec = ~90 m
		self.dy=0.000833

		# Formatting options
		self.savename='Map1'



### ANCILLARY FUNCTIONS ---
## DEFINE AND SAVE MAP - create a gdal georeferenced data set
def buildMap(inpt):
	## Build image
	# Build grids
	inpt.Transform=(inpt.LonStart,inpt.dx,0,inpt.LatStart,0,-inpt.dy)

	x=np.linspace(inpt.LonStart,inpt.LonStart+inpt.N*inpt.dx,inpt.N)
	y=np.linspace(inpt.LatStart,inpt.LatStart+inpt.M*inpt.dy,inpt.M)
	X,Y=np.meshgrid(x,y)
	print(X.shape,Y.shape)

	# Blank image template
	img=np.ones((inpt.M,inpt.N))
	img*=inpt.basevalue

	# Format image
	img=inpt.ImgFcn(img,X,Y)


	## Create and save gdal georeferenced data set
	DS=createDS(inpt,img)

	return DS


## Create data set
def createDS(inpt,img):
	driver=gdal.GetDriverByName('GTiff')
	DS=driver.Create('{}.tif'.format(inpt.savename),inpt.N,inpt.M,1,gdal.GDT_Float32)
	DS.GetRasterBand(1).WriteArray(img)
	DS.SetProjection(inpt.Projection) # DS.SetProjection(ogDS.GetProjection())
	DS.SetGeoTransform(inpt.Transform) # DS.SetGeoTransform(ogDS.GetGeoTransform())
	DS.FlushCache()

	print('Saved to: {}'.format(inpt.savename))

	return DS


## Plot map
def plotMap(inpt,DS):
	## Format map
	# Image
	img=DS.GetRasterBand(1).ReadAsArray()

	# Geo transform
	N=DS.RasterXSize
	M=DS.RasterYSize
	T=DS.GetGeoTransform()
	left=T[0]; dx=T[1]; right=left+N*dx 
	top=T[3]; dy=T[5]; bottom=top+M*dy 
	extent=(left, right, bottom, top)

	## Plot map
	Fig=plt.figure()
	ax=Fig.add_subplot(111)
	cax=ax.imshow(img,extent=extent)
	Fig.colorbar(cax,orientation='horizontal')



### MAIN FUNCTION ---
if __name__=='__main__':
	## Inputs
	inpt=inputs() # instantiate

	# Map parameters
	inpt.M=100 # NS nb pixels
	inpt.N=100 # EW nb pixels

	# Function
	# inpt.ImgFcn=lambda img,X,Y: 20*(X-X.min())*img + 20*(Y-Y.min())*img
	#inpt.ImgFcn=lambda img,X,Y: img
	inpt.ImgFcn=lambda img,X,Y: np.sign(np.sin(2*np.pi*1*(X)/0.000833/100) + np.sin(2*np.pi*1*(Y)/0.000833/100))

	# Other
	inpt.savename='/Users/rzinke/Documents/Tibet/ParameterComparisons/SynthTests/Lumpy2'


	## Create map
	DS=buildMap(inpt)
	plotMap(inpt,DS)

	plt.show()