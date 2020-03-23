#!/usr/bin/env python3
"""
	Plot and regress the second map relative to the first. 


	by Rob Zinke 2019, updated 2020
"""

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from viewingFunctions import imgBackground, mapStats


### PARSER ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Plot and regress the second map relative to the first.')
	# Data sets
	parser.add_argument(dest='baseName', type=str, help='Map to serve as base (independent) data set')
	parser.add_argument(dest='compName', type=str, help='Map to serve as comparison (dependent) to the base')
	# Map options
	parser.add_argument('-b','--bounds', dest='bounds', type=str, default=None, help='Map bounds (minX, minY, maxX, maxY)')
	parser.add_argument('-ds','--downsample','--dsFactor', dest='dsFactor', type=int, default=0, help='Downsample factor (power of two)')
	# Scaling
	parser.add_argument('-cs','--centerscale','--center-scale', dest='centerscale', action='store_true', help='Center and scale data set (True/[False])')
	# Masking
	parser.add_argument('-m','--maskMaps', dest='maskMaps', default=None, nargs='+', help='Georeferenced binary rasters')
	parser.add_argument('-mt','--maskThresholds', dest='maskThresholds', type=float, default=[1], nargs='+', help='Threshold values below which masking will be applied. Use one threshold value per masking map. Default = 1.')
	parser.add_argument('-bg','--background', dest='background', default=None, nargs='+', help='Background value for both maps')
	parser.add_argument('-bgBase','--bgBase', dest='bgBase', default=None, nargs='+', help='Background value for base image')
	parser.add_argument('-bgComp','--bgComp', dest='bgComp', default=None, nargs='+', help='Background value for comparison image')
	# Map plot options
	parser.add_argument('-p','--plotMaps', dest='plotMaps', action='store_true', help='Plot maps')
	parser.add_argument('-c','--cmap','--colormap', dest='cmap', type=str, default='viridis', help='Colormap')
	parser.add_argument('-pctmin','--pctmin', dest='pctmin', type=float, default=0, help='Minimum percent clip')
	parser.add_argument('-pctmax','--pctmax', dest='pctmax', type=float, default=100, help='Maximum percent clip')
	parser.add_argument('--baseLabel', dest='baseLabel', type=str, default='Base', help='Base data set plot name')
	parser.add_argument('--compLabel', dest='compLabel', type=str, default='Comparison', help='Comparison data set plot name')
	# Comparison plot options
	parser.add_argument('-t','--plotType', dest='plotType', type=str, default='pts', help='Comparison plot type ([default = points], )')
	parser.add_argument('-s','--skips', dest='skips', type=int, default=1, help='Skip by this value when plotting points')
	parser.add_argument('--aspect','--plotAspect', dest='plotAspect', type=float, default=None, help='Comparison plot aspect ratio')
	# Analysis options
	parser.add_argument('-a','--analysis','--analysisType', dest='analysisType', type=str, default=None, help='Analysis type (polyfit, PCA, kmeans)')
	parser.add_argument('--degree', dest='degree', type=int, default=1, help='Polynomial degree for fit [default = 1]')
	parser.add_argument('--nbins', dest='nbins', type=int, default=10, help='Number of bins for 2D histograms')
	parser.add_argument('--clusters','--kclusters', dest='kclusters', type=int, default=1, help='Number of clusters for k-means cluster analysis')
	parser.add_argument('--max-iterations', dest='maxIterations', type=int, default=20, help='Max number of iterations for k-means cluster analysis')
	parser.add_argument('--kbins', dest='kbins', type=int, default=20, help='Number of bins to use in data histogram')

	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')

	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name, for difference map and analysis plots')

	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
## Transform to extent
def transform2extent(DS):
	N=DS.RasterXSize; M=DS.RasterYSize
	T=DS.GetGeoTransform()
	left=T[0]; dx=T[1]; right=left+N*dx
	top=T[3]; dy=T[5]; bottom=top+M*dy
	extent=(left, right, bottom, top)

	return extent


## Adjust size and resolution of images
def preFormat(inpt,baseDS,compDS,maskMaps=None):
	baseTnsf=baseDS.GetGeoTransform()

	## First adjust base image to desired extent/resolution
	# Adjust bounds of base image
	if not inpt.bounds:
		N=baseDS.RasterXSize; M=baseDS.RasterYSize
		minX=baseTnsf[0]; maxY=baseTnsf[3]
		dx=baseTnsf[1]; dy=baseTnsf[5]
		maxX=minX+dx*N; minY=maxY+dy*M
		inpt.bounds=(minX, minY, maxX, maxY) # (minX, minY, maxX, maxY)
	else:
		inpt.bounds=eval(inpt.bounds)

	# Adjust resolution of base image
	dsFactor=int(2**inpt.dsFactor)
	dx=dsFactor*baseTnsf[1]; dy=dsFactor*baseTnsf[5]

	# Resample base image
	baseDS=gdal.Warp('',baseDS,options=gdal.WarpOptions(format='MEM',xRes=dx,yRes=dy,
		outputBounds=inpt.bounds,resampleAlg='bilinear'))

	## Then, resample comparison image to match base image
	compDS=gdal.Warp('',compDS,options=gdal.WarpOptions(format='MEM',xRes=dx,yRes=dy,
		outputBounds=inpt.bounds,resampleAlg='bilinear'))

	## Finally, resample mask if specified
	if maskMaps is not None:
		maskMaps=[gdal.Warp('',maskMap,options=gdal.WarpOptions(format='MEM',xRes=dx,yRes=dy,
			outputBounds=inpt.bounds,resampleAlg='near')) for maskMap in maskMaps]

	## Check that geotransforms are equal
	baseTnsf=baseDS.GetGeoTransform(); compTnsf=compDS.GetGeoTransform()
	assert baseTnsf==compTnsf, 'Geo-transforms are not equal!'

	return baseDS, compDS, maskMaps


## Masking by value
def maskByValue(inpt,baseDS,compDS):
	commonMask=np.ones((baseDS.RasterYSize,baseDS.RasterXSize))

	# Determine if background masking is requested
	if (inpt.background is not None) or (inpt.bgBase is not None) or (inpt.bgComp is not None):
		baseImg=baseDS.GetRasterBand(1).ReadAsArray()
		compImg=compDS.GetRasterBand(1).ReadAsArray()

		# Create list of background values
		bgBase=[]
		bgComp=[]


		# Determine global background values
		if inpt.background:
			if 'auto' in inpt.background:
				bgBase.append(imgBackground(baseImg))
				bgComp.append(imgBackground(compImg))
				inpt.background.remove('auto')

			if len(inpt.background)>0:
				[bgBase.append(float(value)) for value in inpt.background]
				[bgComp.append(float(value)) for value in inpt.background]

		# Determine base background values
		if inpt.bgBase:
			if 'auto' in inpt.bgBase:
				bgBase.append(imgBackground(baseImg))
				inpt.bgBase.remove('auto')

			if len(inpt.bgBase)>0:
				[bgBase.append(float(value)) for value in inpt.bgBase]

		# Determine comp background values
		if inpt.bgComp:
			if 'auto' in inpt.bgComp:
				bgComp.append(imgBackground(compImg))
				inpt.bgComp.remove('auto')

			if len(inpt.bgComp)>0:
				[bgComp.append(float(value)) for value in inpt.bgComp]

		# Report if requested
		if inpt.verbose is True:
			print('Background values for BASE: {}'.format(bgBase))
			print('Background values for COMP: {}'.format(bgComp))


		# Update common mask
		for baseValue in bgBase: commonMask[baseImg==baseValue]=0
		for compValue in bgComp: commonMask[compImg==compValue]=0

	return commonMask


## Mask by map
def maskByMap(inpt,maskMaps):
	nMaps=len(maskMaps)
	nThresholds=len(inpt.maskThresholds)

	# Check there is a threshold value for every mask
	if nThresholds<nMaps:
		print('WARNING: Fewer specified masking threshold values than masking maps. Assuming threshold values of 1.')

		# If too few threshold values, fill in with ones
		for n in range(nMaps-nThresholds): inpt.maskThresholds.append(1)

	# Compound common mask by each map
	for n in range(nMaps):
		# Read gdal data set as numpy array
		mask=maskMaps[n].GetRasterBand(1).ReadAsArray()

		# Mask by threshold value
		mask[mask<inpt.maskThresholds[n]]=0
		mask[mask>=inpt.maskThresholds[n]]=1

		inpt.commonMask*=mask

	return mask


## Plot image data set
def plotDatasets(inpt,baseDS,compDS):
	# Geographic formatting
	extent=transform2extent(baseDS)

	# Masking
	baseImg=baseDS.GetRasterBand(1).ReadAsArray()
	baseImg=np.ma.array(baseImg,mask=(inpt.commonMask==0))

	compImg=compDS.GetRasterBand(1).ReadAsArray()
	compImg=np.ma.array(compImg,mask=(inpt.commonMask==0))

	# Min/max values
	baseStats=mapStats(baseImg,pctmin=inpt.pctmin,pctmax=inpt.pctmax)
	compStats=mapStats(compImg,pctmin=inpt.pctmin,pctmax=inpt.pctmax)

	# Plot maps
	Fig=plt.figure()

	axBase=Fig.add_subplot(121) # base figure
	caxBase=axBase.imshow(baseImg,
		vmin=baseStats.vmin,vmax=baseStats.vmax,extent=extent)
	Fig.colorbar(caxBase,orientation='horizontal')
	axBase.set_title(inpt.baseLabel)

	axComp=Fig.add_subplot(122) # comp figure
	caxComp=axComp.imshow(compImg,
		vmin=compStats.vmin,vmax=compStats.vmax,extent=extent)
	axComp.set_yticks([])
	Fig.colorbar(caxComp,orientation='horizontal')
	axComp.set_title(inpt.compLabel)
	Fig.tight_layout()

	# Save if requested
	if inpt.outName:
		savename='{}_side-by-side.png'.format(inpt.outName)
		Fig.savefig(savename,dpi=600)

		# Report if requested
		if inpt.verbose is True:
			print('Saved analysis fig to: {}'.format(savename))



## K-means cluster algorithm
def computeKmeans(data,k,max_iterations):
		# Setup
		dataMin=np.min(data,axis=0)
		dataMax=np.max(data,axis=0)
		dataShape=data.shape
		np.random.seed(0)
		Dists=np.zeros((dataShape[0],k))
		# Pick initial centroids
		Centroids=np.linspace(dataMin,dataMax,k)
		Centroids_new=np.zeros((k,dataShape[1]))
		# Find closest centroids
		def find_closest_centroid(data,Centroids,Dists,k):
			for i in range(k):
				# Subract each centroid from the data
				#  and compute Euclidean distance
				Dists[:,i]=np.linalg.norm(data-Centroids[i,:],
					ord=2,axis=1)
			return Dists
		# Loop through iterations
		while(max_iterations): # for each iteration...
		# 1. Calculate distance to points
			Dists=find_closest_centroid(data,Centroids,Dists,k)
			Centroid_ndx=np.argmin(Dists,axis=1)
		# 2. Assign data points to centroids
			for j in range(k):
				cluster_mean=np.mean(data[Centroid_ndx==j],axis=0)
				Centroids_new[j,:]=cluster_mean
			if not np.sum(Centroids_new-Centroids):
				Centroids=Centroids_new; break
			Centroids=Centroids_new.copy()
			max_iterations-=1

		# List of distances to each centroid's points
		CentroidDists=[]
		Centroid_ndx=np.argmin(Dists,axis=1)
		for i in range(k):
			clusterDists=Dists[Centroid_ndx==i]
			CentroidDists.append(clusterDists[:,i])
		
		return Centroids, CentroidDists



### Compare maps ---
class mapCompare:
	def __init__(self,baseDS,compDS,mask=None,centerscale=False,verbose=False,outName=None):
		self.verbose=verbose # True/False
		self.outName=outName # save files to outName

		# Load image bands and transforms
		baseImg=baseDS.GetRasterBand(1).ReadAsArray()
		compImg=compDS.GetRasterBand(1).ReadAsArray()

		self.N=baseDS.RasterXSize; self.M=baseDS.RasterYSize
		self.Tnsf=baseDS.GetGeoTransform()
		left=self.Tnsf[0]; dx=self.Tnsf[1]; right=left+self.N*dx
		top=self.Tnsf[3]; dy=self.Tnsf[5]; bottom=top+self.M*dy
		self.extent=(left, right, bottom, top)
		self.Projection=baseDS.GetProjection()

		# Mask values
		baseImg=np.ma.array(baseImg,mask=(mask==0))
		compImg=np.ma.array(compImg,mask=(mask==0))

		# Center and scale
		if centerscale is True:
			baseImg=(baseImg-np.mean(baseImg))/np.std(baseImg)
			compImg=(compImg-np.mean(compImg))/np.std(compImg)


		## Simple difference map
		# Compute diff
		self.Diff=baseImg-compImg


		## Compare 1D arrays
		# Flatten to 1D arrays
		baseImg=baseImg.flatten()
		compImg=compImg.flatten()

		# Compress to ignore mask values
		self.base=baseImg.compressed(); del baseImg
		self.comp=compImg.compressed(); del compImg

		self.baseMean=self.base.mean()
		self.compMean=self.comp.mean()

		self.baseStd=self.base.std()
		self.compStd=self.comp.std()

		# Report if requested
		if self.verbose is True:
			print('Base mean {:.4f}; std {:.4f}'.format(self.baseMean,self.baseStd))
			print('Comparison mean {:.4f}; std {:.4f}'.format(self.compMean,self.compStd))


	## Plot difference
	def plotDiff(self,plotProperties,pctmin=0,pctmax=100):
		# Formatting
		vmin,vmax=np.percentile(self.Diff.compressed(),[pctmin,pctmax])

		# Spawn figure
		Fig=plt.figure()
		ax=Fig.add_subplot(111)

		# Plot map
		cax=ax.imshow(self.Diff,cmap='jet',vmin=vmin,vmax=vmax,extent=self.extent)
		ax.set_title('Difference ({} - {})'.format(plotProperties['baseLabel'],plotProperties['compLabel']))
		Fig.colorbar(cax,orientation='horizontal')

		# Save if requested
		if self.outName:
			savename='{}_DifferenceMap.tif'.format(self.outName)
			driver=gdal.GetDriverByName('GTiff')
			DSout=driver.Create(savename,self.N,self.M,1,gdal.GDT_Float32)
			DSout.GetRasterBand(1).WriteArray(self.Diff)
			DSout.SetProjection(self.Projection)
			DSout.SetGeoTransform(self.Tnsf)
			DSout.FlushCache()

			if self.verbose is True:
				print('Saved to: {}'.format(savename))


	## Plot comparison
	def plotComparison(self,plotProperties,analysisProperties):
		# Parse properties
		plotType=plotProperties['plotType']
		analysisType=analysisProperties['analysisType']

		if self.verbose is True:
			print('Plotting comparison type: {}'.format(plotType.upper()))


		# Establish figure
		self.Fig=plt.figure()
		self.ax=self.Fig.add_subplot(111)

		# Plot data
		plotType=plotType.lower()

		if plotType in ['pts','points']:
			self.plotPts(plotProperties)
		elif plotType in ['hex','hexbin']:
			self.plotHex(plotProperties)
		elif plotType in ['hist','hist2d','histogram']:
			self.plotHist(plotProperties)
		elif plotType in ['kde','contour','contourf']:
			self.plotKDE(plotProperties)
		else:
			print('Choose an appropriate plot type'); exit()


		# Conduct and plot analyses
		if analysisType: analysisType=analysisType.lower()

		if analysisType in ['poly','polyfit','polynomial']:
			self.polyFit(analysisProperties)
		elif analysisType in ['pca']:
			self.PCA(analysisProperties)
		elif analysisType in ['kmeans','cluster','clusters','kcluster','kclusters']:
			self.clusterAnalysis(analysisProperties)


		# Finalize figure formatting
		if plotProperties['plotAspect']: self.ax.set_aspect(plotProperties['plotAspect'])
		self.ax.set_xlabel('{} data'.format(plotProperties['baseLabel']))
		self.ax.set_ylabel('{} data'.format(plotProperties['compLabel']))
		self.Fig.tight_layout()

		# Save figure
		if self.outName:
			savename='{}_{}'.format(self.outName,plotType)
			if analysisType: savename+='_{}'.format(analysisType)
			savename+='.png'
			self.Fig.savefig(savename,dpi=600)

			# Report if requested
			if self.verbose is True:
				print('Saved analysis fig to: {}'.format(savename))


	## Data plotting --
	# Points comparison
	def plotPts(self,plotProperties):
		# Parse plotProperties
		skips=plotProperties['skips']
		plotAspect=plotProperties['plotAspect']

		## Plot data points
		self.ax.plot(self.base[::skips],self.comp[::skips],marker='.',color=(0.5,0.5,0.5),linewidth=0,zorder=1)


	# Hexbin comparison
	def plotHex(self,plotProperties):
		# Parse plotProperties
		cmap=plotProperties['cmap']

		## Plot hexbin
		cax=self.ax.hexbin(self.base,self.comp,cmap=cmap)
		self.Fig.colorbar(cax,orientation='horizontal')


	# Histogram comparison
	def plotHist(self,plotProperties):
		# Parse plotProperties
		cmap=plotProperties['cmap']
		nbins=plotProperties['nbins']

		## Construct histogram
		H,xedges,yedges=np.histogram2d(self.base,self.comp,bins=nbins)
		H=H.T
		X,Y=np.meshgrid(xedges,yedges)

		## Plot histogram
		cax=self.ax.pcolormesh(X,Y,H,cmap=cmap)

		self.Fig.colorbar(cax,orientation='horizontal')


	# KDE comparison
	def plotKDE(self,plotProperties):
		# Parse plotProperties
		plotType=plotProperties['plotType'] # plot type for contour contourf
		cmap=plotProperties['cmap']
		nbins=plotProperties['nbins']

		## Compute kernel density estimate
		from scipy.stats import gaussian_kde

		x=np.linspace(self.base.min(),self.base.max(),nbins)
		y=np.linspace(self.comp.min(),self.comp.max(),nbins)

		X,Y=np.meshgrid(x,y)
		positions=np.vstack([X.flatten(),Y.flatten()])
		values=np.vstack([self.base.flatten(),self.comp.flatten()])
		kernel=gaussian_kde(values)
		H=kernel(positions)
		H=np.reshape(H,X.shape)

		## Plot KDE
		if plotType in ['kde']:
			# Plot heat map
			cax=self.ax.pcolormesh(X,Y,H,cmap=cmap)
		elif plotType in ['contour']:
			# Plot contours on top of point measurements
			skips=plotProperties['skips']
			self.ax.plot(self.base[::skips],self.comp[::skips],marker='.',color=(0.5,0.5,0.5),linewidth=0,zorder=1)
			cax=self.ax.contour(X,Y,H,cmap=cmap)
		elif plotType in ['contourf']:
			# Plot filled contours
			cax=self.ax.contourf(X,Y,H,cmap=cmap)
		self.Fig.colorbar(cax,orientation='horizontal')


	## Analyses --
	# Polynomial fit
	def polyFit(self,analysisProperties):
		## Fit polynomial
		# Determine degree of polynomial
		degree=analysisProperties['degree']

		# Fit polynomial
		fit=np.polyfit(self.base,self.comp,deg=degree)

		## Plot polynomial
		# Evaluate polynomial at x-coordinates
		P=np.poly1d(fit)
		self.comp_hat=P(self.base) # predicted values

		# Compute statistics
		n=len(self.base)
		m=degree+1

		SStot=np.sum((self.comp-self.comp.mean())**2)
		SSres=np.sum((self.comp-self.comp_hat)**2)

		Rsquared=1-SSres/SStot

		MSE=np.sum((self.comp-self.comp_hat)**2)/(n-m)

		# Report if requested
		if self.verbose is True:
			print('Fitting degree: {}'.format(degree))
			[print('x^{}: {}'.format(degree-n,coeff)) for n,coeff in enumerate(fit)]
			print('Rsquared: {}'.format(Rsquared))
			print('MSE: {}'.format(MSE))

		## Plot polynomial
		# Plot polynomial fit
		ndx=np.argsort(self.base)
		self.ax.plot(self.base[ndx],self.comp_hat[ndx],'k--')

		# Text box
		textstr='Linear regression analysis'
		textstr+='\nRelationship:'
		for n,coeff in enumerate(fit):
			textstr+=' {:.3f} $x^{}$'.format(coeff,degree-n)
		textstr+='\n$R^2$: {:.4f}'.format(Rsquared)
		self.ax.text(0.05,0.85,textstr,transform=self.ax.transAxes)


	# PCA
	def PCA(self,analysisProperties):
		## PCA analysis
		# Note -- no analysisProperties accepted for now

		# Reformat in data matrix
		A=np.hstack([self.base.reshape(-1,1),self.comp.reshape(-1,1)])

		# Remove centroid
		A[:,0]=A[:,0]-self.baseMean
		A[:,1]=A[:,1]-self.compMean

		# Covariance matrix
		Cov=np.dot(A.T,A)/(A.shape[0]-1)

		# Eigen decomp
		eigvals,eigvecs=np.linalg.eig(Cov)
		sort_ndx=np.argsort(eigvals)[::-1] # reorder
		eigvals=eigvals[sort_ndx]
		eigvecs=eigvecs[:,sort_ndx]

		# Standard deviations
		eigstds=np.sqrt(eigvals)

		# Construct principal components
		PC1=eigstds[0]*eigvecs[:,0]
		PC2=eigstds[1]*eigvecs[:,1]

		# Percent variance
		totalVar=np.sum(eigvals) # sum of eigenvalues
		pctVar=100*eigvals/totalVar # percent variance explained

		# Relationship between base and comp
		eigslope=eigvecs[1,0]/eigvecs[0,0]

		# Report if requested
		if self.verbose is True:
			print('Eigenvalues: {}'.format(eigvals))
			print('Eigenvectors:\n{}'.format(eigvecs))


		## Plot PCA
		# Plot data centroid
		self.ax.plot(self.baseMean,self.compMean,'ko',zorder=2)

		# Plot eigenbasis
		scale=3 # standard deviations
		self.ax.plot([-scale*PC1[0]+self.baseMean,scale*PC1[0]+self.baseMean],\
			[-scale*PC1[1]+self.compMean,scale*PC1[1]+self.compMean],'k--')
		self.ax.plot([-scale*PC2[0]+self.baseMean,scale*PC2[0]+self.baseMean],\
			[-scale*PC2[1]+self.compMean,scale*PC2[1]+self.compMean],'k--')

		# Text box
		textstr='Principal component analysis'
		textstr+='\nRelationship: {:.3f}\nPC1: {:.3f}\nPC2: {:.3f}'.format(eigslope,pctVar[0],pctVar[1])
		self.ax.text(0.05,0.85,textstr,transform=self.ax.transAxes)


	# K-means cluster analysis
	def clusterAnalysis(self,analysisProperties):
		## K-clusters
		# Parse analysis properties
		k=analysisProperties['kclusters']
		max_iterations=analysisProperties['maxIterations']
		nbins=analysisProperties['nbins']

		# Format data
		data=np.hstack([self.base.reshape(-1,1),self.comp.reshape(-1,1)])

		# Compute kmeans
		centroids,centroidDists=computeKmeans(data,k,max_iterations)

		# Report if requested
		if self.verbose is True:
			print('{} clusters computed'.format(k))
			print('Centroids:\n')
			[print('\t{}'.format(centroid)) for centroid in centroids]


		## Plot cluster centers
		for centroid in centroids:
			self.ax.plot(centroid[0],centroid[1],'bo')

		# Plot distance distribution to each centroid
		percentiles=[68,95,98]

		Fig=plt.figure()
		for i in range(k):
			# Compute histogram for distances to each centroid
			H,Hedges=np.histogram(centroidDists[i],bins=nbins)
			Hcntrs=(Hedges[:-1]+Hedges[1:])/2 

			# Pad arrays for plotting
			Hcntrs=np.pad(Hcntrs,(1,1),'edge')
			H=np.pad(H,(1,1),'constant')

			# Plot distance histogram
			ax=Fig.add_subplot(k,1,i+1)
			ax.fill(Hcntrs,H,'b',zorder=1)
			ax.set_yticks([]); ax.set_ylabel('freq')
			ax.text(0.75,0.85,'Centroid {}'.format(i),transform=ax.transAxes)

			# Compute and plot percentiles
			pcts=np.percentile(centroidDists[i],percentiles)
			for j,pct in enumerate(pcts):
				ax.axvline(pct,color=(0.6,0.6,0.6),zorder=2)
				ax.text(pct,0,percentiles[j],color=(0.9,0.5,0.0))
		ax.set_xlabel('distance from centroid')



### MAIN FUNCTION ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	## Load maps
	# Load base data set
	baseDS=gdal.Open(inpt.baseName,gdal.GA_ReadOnly)

	# Load comparison data set
	compDS=gdal.Open(inpt.compName,gdal.GA_ReadOnly)

	# Load mask if specified
	if inpt.maskMaps:
		# Open each mask data set and store as a list
		maskMapDSs=[gdal.Open(maskMap,gdal.GA_ReadOnly) for maskMap in inpt.maskMaps]
	else:
		maskMapDSs=None


	## Pre-format data sets - sample to same map bounds and resolution
	baseDS,compDS,maskMaps=preFormat(inpt,baseDS,compDS,maskMapDSs)


	## Mask by map and/or values
	# Mask by value(s)
	inpt.commonMask=maskByValue(inpt,baseDS,compDS)

	# Mask by map
	if inpt.maskMaps:
		maskByMap(inpt,maskMaps)


	## Plot input images
	if inpt.plotMaps is True:
		plotDatasets(inpt,baseDS,compDS)


	## Compare two data sets
	# Format plot properties
	plotProperties=dict(baseLabel=inpt.baseLabel,compLabel=inpt.compLabel,plotType=inpt.plotType,\
		skips=inpt.skips,plotAspect=inpt.plotAspect,cmap=inpt.cmap,nbins=inpt.nbins)

	# Format analysis properties
	analysisProperties=dict(analysisType=inpt.analysisType,degree=inpt.degree,kclusters=inpt.kclusters,maxIterations=inpt.maxIterations,nbins=inpt.kbins)

	# Conduct comparison
	comparison=mapCompare(baseDS,compDS,mask=inpt.commonMask,centerscale=inpt.centerscale,verbose=inpt.verbose,outName=inpt.outName)
	comparison.plotDiff(plotProperties=plotProperties,pctmin=inpt.pctmin,pctmax=inpt.pctmax)
	comparison.plotComparison(plotProperties=plotProperties,analysisProperties=analysisProperties)


	plt.show()