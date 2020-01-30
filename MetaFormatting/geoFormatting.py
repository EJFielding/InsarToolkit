# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Format and transform geographic data
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
import numpy as np 

# --- Convert pixel location to map location ---
# Pixels to map coordinates
def px2coords(tnsf,px,py):
	left=tnsf[0]; dx=tnsf[1]
	top=tnsf[3]; dy=tnsf[5]
	xcoord=left+px*dx
	ycoord=top+py*dy
	return xcoord, ycoord

# Map coordinates to pixels
def coords2px(tnsf,lon,lat):
	left=tnsf[0]; dx=tnsf[1]
	top=tnsf[3]; dy=tnsf[5]
	px=int((lon-left)/dx)
	py=int((top-lat)/dy)
	return px,py


# --- Simple map extent ---
# Convert geo transform to extent
def transform2extent(tnsf,M,N):
	left=tnsf[0]; dx=tnsf[1]; right=left+dx*N
	top=tnsf[3]; dy=tnsf[5]; bottom=top+dy*M
	extent=(left,right,bottom,top)
	return extent

# --- GDAL geographic transform --- 
# Format transform data into something useful 
class GDALtransform:
	def __init__(self,DS=None,transform=None,shape=None,vocal=False):
		from osgeo import gdal
		# transform comes from data.GetGeoTransform() 
		# shape comes from data.GetRasterBand(#).ReadAsArray().shape 
		if DS is not None:
			transform=DS.GetGeoTransform() 
			shape=(DS.RasterYSize,DS.RasterXSize) 
		self.m=shape[0] 
		self.n=shape[1] 
		self.transform=transform 
		self.xstart=transform[0]
		self.ystart=transform[3]
		self.ystep=transform[5]
		self.xstep=transform[1]
		self.xend=self.xstart+shape[1]*self.xstep 
		self.yend=self.ystart+shape[0]*self.ystep 
		self.ymin=np.min([self.yend,self.ystart])
		self.ymax=np.max([self.yend,self.ystart])
		self.xmin=np.min([self.xend,self.xstart])
		self.xmax=np.max([self.xend,self.xstart]) 
		self.extent=[self.xmin,self.xmax,self.ymin,self.ymax] 
		self.bounds=[self.xmin,self.ymin,self.xmax,self.ymax] 
		# Print outputs? 
		if vocal is not False: 
			print('Image properties: ')
			print('\tNS-dim (m): %i' % self.m) 
			print('\tEW-dim (n): %i' % self.n) 
			print('\tystart: %f\tyend: %f' % (self.ystart,self.yend)) 
			print('\txstart: %f\txend: %f' % (self.xstart,self.xend)) 
			print('\tystep: %f\txstep: %f' % (self.ystep,self.xstep))