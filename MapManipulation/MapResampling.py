# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions for resampling geospatial data  
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from osgeo import gdal 

# --- Resample using gdal Warp --- 
# INPUTS 
#	DS is the original georeferenced image 
#	format is the output format ["MEM","VRT"]
#	bounds are map bounds in format (xmin, ymin, xmax, ymax) 
#	xRes is the x/Lon resolution [deg/m] 
#	yRes is the y/Lat resolution [deg/m] 
#	outType is the output data format [gdal.GDT_... Float32, Int16, etc.] 
#	resampleAlg is the resampling algorithm [lanczos,near,bi] 
#	dstNoData is the destination no data value 
# OUTPUTS 
#	R is the resampled map 
def GDALresample(DS,format,bounds,xRes,yRes,outType=gdal.GDT_Float32,resampleAlg='lanczos',dstNoData=0): 
	# Resample imagery 
	R=gdal.Warp('',DS,\
		options=gdal.WarpOptions(format="MEM",outputType=outType,
			outputBounds=bounds,xRes=xRes,yRes=yRes,
			resampleAlg=resampleAlg,
			dstNodata=dstNoData,srcNodata=0)) 
	return R 



