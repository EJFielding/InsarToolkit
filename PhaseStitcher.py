# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stitch unwrapped phase based on areas of 
#  overlap between frames 
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Standard modules 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats 
from osgeo import gdal 
# RZ modules 
from InsarFormatting import * 
from ConnCompHandler import * 

class ARIAphaseStitch: 
	# Provide a list of files to stitch together 
	def __init__(self,Flist,outDir,vocal=False): 
		nF=len(Flist) # number of files to read 
		# Use only "filename" part of file paths 
		Fnames=[] # empty list of file names 
		for i in range(nF): 
			name=os.path.basename(Flist[i]) # use only file basename 
			name=name.strip('\n'); name=name.strip('.nc') # remove suffixes 
			Fnames.append(name) # add to list of filenames  
		# Establish empty parameters 
		PHSlist=[] # empty list of phase datasets 
		CMPlist=[] # empty list of connected components datasets 
		PhsList=[] # empty list of phase maps 
		CmpList=[] # empty list of connected components maps 
		xMin=[]; xMax=[] # empty list of x/Lon limits 
		yMin=[]; yMax=[] # empty list of y/Lat limits 
		xstep=[]; ystep=[] # empty list of pixel step sizes 
		# 1. Load in images using gdal 
		for i in range(nF): 
			# 1.1 Load phase 
			PHS=gdal.Open('NETCDF:"%s":/science/grids/data/unwrappedPhase' \
				% Flist[i],gdal.GA_ReadOnly) 
			PHSlist.append(PHS) # add to list of phase datasets 

			# 1.2 Load connected components 
			CMP=gdal.Open('NETCDF:"%s":/science/grids/data/connectedComponents' \
				% Flist[i],gdal.GA_ReadOnly) 
			CMPlist.append(CMP) # add to list of connected components datasets 

			# 1.3 Record geographic parameters 
			tnsf=PHS.GetGeoTransform() # gdal geographic transform 
			# (0 xstart, 1 xstep, 2 xshear, 3 ystart, 4 yshear, 5 ystep) 
			xMin.append(tnsf[0]); xMax.append(tnsf[0]+tnsf[1]*PHS.RasterXSize) 
			yMax.append(tnsf[3]); yMin.append(tnsf[3]+tnsf[5]*PHS.RasterYSize) 
			xstep.append(tnsf[1]); ystep.append(np.abs(tnsf[5])) 

			# Vocalize if requested 
			if vocal is True: 
				print('Loaded: %s' % (Fnames[i])) 
				print('\tGeographic parameters') 
				print('\t\txMin: %f\txMax: %f' % (xMin[i],xMax[i])) 
				print('\t\tyMin: %f\tyMax: %f' % (yMin[i],yMax[i])) 
				print('\t\txstep: %f\tystep: %f' % (xstep[i],ystep[i])) 


		# 2. Determine spatial extent/resolution 
		# 2.1 Determine spatial extent/resolution 
		xMin=min(xMin); xMax=max(xMax) # x/Lon extent 
		yMin=min(yMin); yMax=max(yMax) # y/Lat extent 
		bounds=(xMin,yMin,xMax,yMax)   # full map bounds in gdal format 
		xstep=max(xstep); ystep=max(ystep) # x,y resolution 
		self.extent=(xMin,xMax,yMin,yMax) # record geographic extent  
		if vocal is True: 
			print('Geographic extent:') 
			print('\txMin: %f\txMax: %f' % (xMin,xMax)) 
			print('\tyMin: %f\tyMax: %f' % (yMin,yMax)) 
			print('\txstep: %f\tystep: %f' % (xstep,ystep)) 

		# 2.2 Resample onto common grid 
		for i in range(nF): 
			# Resample phase with Lanczos/sinc algorithm 
			PHSlist[i]=gdal.Warp(outDir+Fnames[i]+'UNW.vrt',PHSlist[i],options=gdal.WarpOptions(
				format="VRT",outputType=gdal.GDT_Float32,
				outputBounds=bounds,xRes=xstep,yRes=ystep,
				resampleAlg='lanczos',dstNodata=0,srcNodata=0)) 
			# Resample conn comp with nearest neighbor algorithm 
			CMPlist[i]=gdal.Warp(outDir+Fnames[i]+'CMP.vrt',CMPlist[i],options=gdal.WarpOptions(
				format="VRT",outputType=gdal.GDT_Int16,
				outputBounds=bounds,xRes=xstep,yRes=ystep,
				resampleAlg='near',dstNodata=0,srcNodata=0)) 

			if vocal is True: 
				print('Resampling: %s' % (Flist[i]))

		# 3. Use connected components as a mask 
		for i in range(nF): 
			# Draw maps from resampled dataset 
			Phs=PHSlist[i].GetRasterBand(1).ReadAsArray() 
			Cmp=CMPlist[i].GetRasterBand(1).ReadAsArray() 

			# 3.1 Reorder connected components 
			Cmp=cmpOrder(Cmp,BG=0) 

			# 3.2 Extract primary component of phase map 
			PhsMap,CmpMap=cmpIsolate(Phs,Cmp,cmpList=[1],NoData_out='mask') # assign nodata as 0 
			Phs=PhsMap[1]; Cmp=CmpMap[1] # use primary component 

			# Append map lists 
			PhsList.append(Phs) 
			CmpList.append(Cmp) 

			# Vocalize if requested 
			if vocal is True: 
				print('Adjusting conn comp: %s' % (Fnames[i])) 
				print('\treordered components') 
				print('\tmasking secondary components') 

		# 4. Remove difference in area of overlap 
		#  Work pairwise between overlapping images 
		nPairs=nF-1 # number of adjacent pairs 
		PhsFull=PhsList[0].copy() # starting values for stitched phase 
		PhsFull.mask=False # unmask 
		CmpFull=CmpList[0].copy() # starting values for stitched comps 
		CmpFull.mask=False # unmask 
		for i in range(nPairs): 
			Phs1=PhsList[i]; Phs2=PhsList[i+1] # temporary phase maps 
			Cmp1=CmpList[i]; Cmp2=CmpList[i+1] # temporary comp maps 

			# 4.1 Find area of overlap 	
			OLmask=np.zeros(CmpList[i].shape) # empty mask based on map size 
			OLmask[(CmpList[i]==1) & (CmpList[i+1]==1)]=1 # pair overlap values 

			# 4.2 Compute difference between frames based on overlap 
			Diff=Phs2-Phs1 # compute difference everywhere 

			# Difference stats 
			Darray=Diff.reshape(1,-1).squeeze(0) # reshape to 1D array 
			Darray=Darray.compressed() # use only non-masked values 

			Dmean=np.mean(Darray)       # mean 
			Dmedian=np.median(Darray)   # median 
			Dmode=stats.mode(Darray)[0] # mode 
			Dstdev=np.std(Darray)       # standard deviation

			# 4.3 Remove difference from second image 
			# Subtract mean difference 
			PhsList[i+1]=PhsList[i+1]-Dmean # from original phase map 
			Phs2=Phs2-Dmean # from temporary phase map 

			# Recompute stats 
			Diff2=Phs2-Phs1 # corrected difference 
			D2=Diff2.reshape(1,-1).squeeze(0) # reshape to 1D array 
			D2=D2.compressed() # use only non-masked values 

			D2mean=np.mean(D2) # mean of residuals 
			D2rms=np.sqrt(np.sum(D2**2)/len(D2)) # RMS of residuals 

			# 4.4 Stitch data 
			# Unmask temporary data 
			Phs2.mask=False 
			Cmp2.mask=False 

			# Combine data 
			PhsFull=PhsFull+Phs2  
			CmpFull=CmpFull+Cmp2 

			# Deal with overlaps 
			PhsFull[OLmask==1]=PhsFull[OLmask==1]/2 
			CmpFull[OLmask==1]=CmpFull[OLmask==1]/2 

			# Vocalize if requested 
			if vocal is True: 
				print('Calculating difference between phases') 
				print('\tDifference stats') 
				print('\tmean: %f\n\tmedian: %f\n\tmode: %f\n\tstdev: %f' % \
					(Dmean,Dmedian,Dmode,Dstdev)) 
				print('Removing difference') 
				print('\tresidual mean: %f\n\tRMS residual: %f' % \
					(D2mean,D2rms)) 
				print('Stiched: %s to full map' % (Fnames[i]))

		# Remask secondary components 
		PhsMap,CmpMap=cmpIsolate(PhsFull,CmpFull,cmpList=[1],NoData_out=0) 
		self.StitchedPhase=PhsMap[1] # save final phase map 
		self.StitchedComps=CmpMap[1] # save final comps map 

		# Report if requested 
		if vocal is True: 
			print('Stitching complete.')

		# 5. Save/Output 
		m,n=self.StitchedPhase.shape 
		proj=PHSlist[0].GetProjection() 
		driver=gdal.GetDriverByName('GTiff') 
		PHSfull=driver.Create(outDir+'StitchedPhase',n,m,1,gdal.GDT_Float32) 
		PHSfull.GetRasterBand(1).WriteArray(self.StitchedPhase) 
		PHSfull.SetProjection(proj) 
		PHSfull.SetGeoTransform(tnsf) 
		PHSfull.FlushCache() 

		CMPfull=driver.Create(outDir+'StitchedComps',n,m,1,gdal.GDT_Int16) 
		CMPfull.GetRasterBand(1).WriteArray(self.StitchedComps) 
		CMPfull.SetProjection(proj) 
		CMPfull.SetGeoTransform(tnsf) 
		CMPfull.FlushCache() 

	# Plot if specified 
	def plot_stitched(self): 
		Phs=np.ma.array(self.StitchedPhase,mask=(self.StitchedComps==0)) 
		Cmp=np.ma.array(self.StitchedComps,mask=(self.StitchedComps==0))  
		F=plt.figure() 
		ax1=F.add_subplot(1,2,1) 
		cax1=ax1.imshow(Phs,extent=self.extent) 
		ax1.set_title('Phase')
		F.colorbar(cax1,orientation='vertical') 
		ax2=F.add_subplot(1,2,2) 
		cax2=ax2.imshow(Cmp,cmap='nipy_spectral',extent=self.extent) 
		ax2.set_title('Components') 
		F.colorbar(cax2,orientation='vertical') 




