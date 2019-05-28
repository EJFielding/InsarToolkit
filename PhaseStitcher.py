# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stitch unwrapped phase based on areas of 
#  overlap between frames 
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 
from InsarFormatting import * 
from ConnCompHandler import * 

class ARIAphaseStitch: 
	# 1. Provide a list of files to stitch together 
	def __init__(self,Flist): 
		nF=len(Flist) # number of files to read 
		# Establish empty parameters 
		CmpList=[] # empty list of connected components files 
		PhsList=[] # empty list of phase files 
		xMin=[]; xMax=[] # empty list of x/Lon limits 
		yMin=[]; yMax=[] # empty list of y/Lat limits 
		# 1.1 Load in images using gdal 
		for i in range(nF): 
			# 1.1.1 Load phase 
			PHS=gdal.Open('NETCDF:"%s":/science/grids/data/unwrappedPhase' \
				% Flist[i],gdal.GA_ReadOnly) 
			Phs=PHS.GetRasterBand(1).ReadAsArray() 

			# 1.1.2 Load connected components 
			CMP=gdal.Open('NETCDF:"%s":/science/grids/data/connectedComponents' \
				% Flist[i],gdal.GA_ReadOnly) 
			Cmp=CMP.GetRasterBand(1).ReadAsArray() 

			# 1.1.3 Reorder connected components 
			Cmp=cmpOrder(Cmp,BG=0,hist=True) 

			# 1.1.4 Extract primary component of phase map 

			# 1.1.5 Record geographic parameters 

	# 2. Resample on exact same grid 

	# 3. Compute difference between frames based on overlap 

	# 4. Remove difference 

	# 5. Save/Output 




