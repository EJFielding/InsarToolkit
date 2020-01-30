#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from osgeo import gdal
from viewingFunctions import mapPlot
from geoFormatting import transform2extent, GDALtransform

### --- Parser ---
def createParser():
	'''
		Create full-resolution maps of tide data.
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Create full resolution maps of tide data')
	parser.add_argument('-i','--IFGfile', dest='IFGfile', type=str, required=True, help='IFG file for spatial reference')
	parser.add_argument('-d','--dateList', dest='dateListFile', type=str, required=True, help='List of dates')
	parser.add_argument('-l','--lookupTable', dest='lookupTableFile', type=str, default='.', help='Table associating dates with snaps')
	parser.add_argument('-s','--snapDir', dest='snapDir', type=str, required=True, help='Directory with snap files')
	parser.add_argument('-o','--outDir', dest='outDir', type=str, default=None, help='Directory to save images')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-p','--plot', dest='plot', action='store_true', help='Plot')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Processing functions ---
## Convert Snap file to CSV readable by gdal
def Snap2CSV(inpt,Lon,Lat,data,CSVname):
	XYZ=np.vstack([Lon,Lat,data]).T # reformat tide data
	np.savetxt(CSVname,XYZ,fmt='%.4f',delimiter=',') # save as comma-delimited
	if inpt.verbose is True:
		print('Saved CSV: {}'.format(CSVname))

## Convert CSV to TIFF and adjust coorindate system
def CSV2Tiff(inpt,CSVname,TIFname,mapProjection):
	# Convert to tiff using gdal translate
	#  This image likely has an improper coordinate system
	TideMap=gdal.Translate('',CSVname, options=gdal.TranslateOptions(format='MEM',outputType=gdal.GDT_Float32))
	tideField=TideMap.GetRasterBand(1).ReadAsArray() # array of tide displacement values
	M=TideMap.RasterYSize; N=TideMap.RasterXSize # map dimensions
	GeoTnsf=TideMap.GetGeoTransform(); GeoTnsf=list(GeoTnsf) # geo transform
	if inpt.verbose is True:
		print('Saving csv: {} to tiff: {}'.format(CSVname,TIFname))
		print('\tOrig geotransform: {}'.format(GeoTnsf))

	# Correct coordinate system
	tideField=np.flipud(tideField) # flip array of tide displacement values top for bottom
	GeoTnsf[3]=GeoTnsf[3]+M*GeoTnsf[5] # calculate northernmost coordinate
	GeoTnsf[5]=-GeoTnsf[5] # use negative y-value
	if inpt.verbose is True:
		print('\tFlipped map top for bottom')
		print('\tUsing northern coordinate: {}'.format(GeoTnsf[3]))
		print('\tUsing negative Y value: {}'.format(GeoTnsf[5]))
		print('\tReformulated geotransform: {}'.format(GeoTnsf))

	# Save to file
	Driver=gdal.GetDriverByName('GTiff')
	OutMap=Driver.Create(TIFname,N,M,1,gdal.GDT_Float32)
	OutMap.GetRasterBand(1).WriteArray(tideField)
	OutMap.SetProjection(mapProjection)
	OutMap.SetGeoTransform(GeoTnsf)
	OutMap.FlushCache() 


## Plot thumbnails
def plotTide(F,i,field,TideMap):
	# Gather image data
	tideField=TideMap.GetRasterBand(1).ReadAsArray()
	M=TideMap.RasterYSize; N=TideMap.RasterXSize
	geoTnsf=TideMap.GetGeoTransform()
	extent=transform2extent(geoTnsf,M,N)

	# Plot imagette
	ax=F.add_subplot(2,2,i)
	cax=ax.imshow(tideField,cmap='jet',extent=extent)
	ax.set_title(field)
	F.colorbar(cax,orientation='vertical')


### --- Main function ---
if __name__ == '__main__':
	# Gather inputs
	inpt=cmdParser()

	if not inpt.outDir:
		# Unless specified, save tide maps to 
		#  directory with snap files
		inpt.outDir=inpt.snapDir

	## Load single interferogram for spatial reference
	IFG=gdal.Open(inpt.IFGfile,gdal.GA_ReadOnly)
	ifg=IFG.GetRasterBand(1).ReadAsArray()
	M,N=ifg.shape
	T=GDALtransform(IFG)
	transform=IFG.GetGeoTransform()
	extent=transform2extent(transform,M,N)
	ReferenceProjection=IFG.GetProjection()
	if inpt.verbose is True:
		print('Reference projection: {}'.format(ReferenceProjection))

	# Plot if requested
	if inpt.plot is True:
		F=plt.figure()
		ax=F.add_subplot(221)
		cax=ax.imshow(ifg,extent=extent)
		F.colorbar(cax,orientation='vertical')


	## Load list of each date in ifg data set
	# Load date list
	with open(inpt.dateListFile,'r') as dateListFile:
		dateList=dateListFile.readlines()
		dateListFile.close()
	# Format date list
	dateList=[date.strip('\n') for date in dateList]
	nDates=len(dateList)
	
	if inpt.verbose is True:
		print('Dates: {}'.format(dateList))
		print('N dates: {}'.format(nDates))
	

	## Load lookup table
	# Load lookup table
	with open(inpt.lookupTableFile,'r') as lookupTableFile:
		lookupTableList=lookupTableFile.readlines()
		lookupTableFile.close()
	# Format lookup table
	lookupTable={}
	for entry in lookupTableList:
		entry=entry.strip('\n')
		date,snap=entry.split(',')
		lookupTable[date]=snap # append to dictionary
		#if inpt.verbose is True:
		#	print('Date {}, Snap {}'.format(date,snap))


	## Create maps of tidal effects
	# Loop through each date
	for date in dateList:

		# Load snap file corresponding to date
		snapBasename=lookupTable[date]
		if inpt.verbose is True:
			print('Date {}, Snap {}'.format(date,snapBasename))
		
		# Convert text data for displacements to numpy arrays
		snapFilename=os.path.join(inpt.snapDir,'{}.txt'.format(snapBasename))
		snapData=np.loadtxt(snapFilename) # load existing data
		Lat=snapData[:,0]; Lon=snapData[:,1]
		DisplacementFields={}
		DisplacementFields['Vert']=snapData[:,2]
		DisplacementFields['EW']=snapData[:,3]
		DisplacementFields['NS']=snapData[:,4]

		# Save data as gdal-compatible txt format
		#  One csv file per displacement field (Vert, EW, NS)
		CSVnames=[]
		for field in DisplacementFields.keys():
			# Formulate CSV name
			CSVname=os.path.join(inpt.snapDir,'{}{}.csv'.format(date,field))
			CSVnames.append(CSVname) # add name to list
			# Save to XYZ
			Snap2CSV(inpt,Lon,Lat,DisplacementFields[field],CSVname)

		# Convert CSV to TIFF and adjust coordinate system
		#  One img file per displacement field (Vert, EW, NS)
		TIFnames=[]
		F=plt.figure(); i=1
		for field,CSVname in zip(DisplacementFields.keys(), CSVnames):
			# Formulate Tiff name
			TIFname=os.path.join(inpt.outDir,'{}{}.tif'.format(date,field))
			TIFnames.append(TIFname) # add name to list
			# Convert, correct spatial reference, and save to TIFF
			CSV2Tiff(inpt,CSVname,TIFname,ReferenceProjection)
			# Load corrected data
			TideMap=gdal.Open(TIFname,gdal.GA_ReadOnly)
			# Plot and save imagettes
			plotTide(F,i,field,TideMap)			
			i+=1
		# Save thumbnail plot for reference
		F.suptitle('{} ocean tide displacements (mm)'.format(date))
		F.savefig('{}/{}_tide_data'.format(inpt.outDir,date))

	if inpt.plot is True:
		plt.show()



