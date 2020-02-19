import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import mode


##########################
### --- STATISTICS --- ###
##########################

# --- Image background ---
def imgBackground(I):
	# Use mode of background values
	edgeValues=np.concatenate([I[0,:],I[-1,:],I[:,0],I[:,-1]])
	background=mode(edgeValues).mode[0] # most common value
	return background

# --- Map stats ---
class mapStats:
	def __init__(self,I,pctmin=0,pctmax=100,verbose=False,hist=False): 
		# Guess at background value
		try:
			self.background=imgBackground(I)
		except:
			self.background=None
		# Check if masked array 
		try: 
			I=I.compressed() 
		except: 
			pass 
		# Convert to 1D array
		I=np.reshape(I,(1,-1)).squeeze(0) # 1D array 
		# Stats 
		self.min=np.min(I)	     # min 
		self.max=np.max(I)	     # max 
		self.mean=np.mean(I)	 # mean 
		self.median=np.median(I) # median 
		self.std=np.std(I)       # standard deviation 
		self.vmin,self.vmax=np.percentile(I,(pctmin,pctmax)) 
		# Print stats 
		if verbose is True: 
			print('Image stats:')
			print('\tmin: {}, max: {}'.format(self.min,self.max)) 
			print('\tmean: {}'.format(self.mean)) 
			print('\tmedian: {}'.format(self.median)) 
			print('\tvmin: {}, vmax: {}'.format(self.vmin,self.vmax)) 
			print('\tlikely background: {}'.format(self.background))
		# Histogram 
		if hist is not False: 
			if type(hist)==int: 
				nbins=hist 
			else: 
				nbins=50 
			# All values 
			H0,H0edges=np.histogram(I,bins=nbins) 
			H0cntrs=H0edges[:-1]+np.diff(H0edges)/2 
			# Clipped values 
			I=I[(I>=self.vmin) & (I<=self.vmax)] 
			H,Hedges=np.histogram(I,bins=nbins) 
			Hcntrs=Hedges[:-1]+np.diff(Hedges)/2 
			# Plot 
			plt.figure() 
			# Plot CDF 
			plt.subplot(2,1,1) 
			plt.axhline(pctmin/100,color=(0.5,0.5,0.5))
			plt.axhline(pctmax/100,color=(0.5,0.5,0.5)) 
			plt.plot(H0cntrs,np.cumsum(H0)/np.sum(H0),'k') 
			# Pad 
			H0cntrs=np.pad(H0cntrs,(1,1),'edge')
			H0=np.pad(H0,(1,1),'constant') 
			Hcntrs=np.pad(Hcntrs,(1,1),'edge')
			H=np.pad(H,(1,1),'constant') 
			# Plot PDF 
			plt.subplot(2,1,2)  
			plt.fill(H0cntrs,H0,color=(0.4,0.5,0.5),alpha=1,label='orig') 
			plt.bar(Hcntrs,H,color='r',alpha=0.5,label='new') 
			plt.legend() 


###########################
### --- Map viewing --- ###
###########################

# --- Plot single-band image ---
def mapPlot(img,cmap='viridis',vmin=None,vmax=None,pctmin=None,pctmax=None,background=None,
	extent=None,showExtent=False,cbar_orientation=None,title=None):
	# Create figure
	F=plt.figure()
	ax=F.add_subplot(111)

	# Format image
	if pctmin is not None or pctmax is not None:
		assert vmin is None and vmax is None, 'Specify either vmin/max or pctmin/max, not both'
		stats=mapStats(img,pctmin=pctmin,pctmax=pctmax); vmin=stats.vmin; vmax=stats.vmax 

	if background is not None:
		if background=='auto':
			background=imgBackground(img)
		img=np.ma.array(img,mask=(img==background))

	# Plot image
	cax=ax.imshow(img,extent=extent,
		cmap=cmap,vmin=vmin,vmax=vmax)

	# Plot formatting
	if cbar_orientation:
		F.colorbar(cax,orientation=cbar_orientation)
	if showExtent is False:
		ax.set_xticks([]); ax.set_yticks([])
	if title:
		ax.set_title(title)

	return F,ax


# --- Plot overlay image ---
def overlayMap(Base,Overlay,BaseCmap='Greys_r',OverlayCmap='viridis',OverlayAlpha=0.5,BaseVmin=None,BaseVmax=None,OverlayVmin=None,OverlayVmax=None,
	extent=None,cbar_orientation='horizontal',title=None):
	# Create figure
	F=plt.figure()
	ax=F.add_subplot(111)

	# Base image
	ax.imshow(Base,extent=extent,
		cmap=BaseCmap,vmin=BaseVmin,vmax=BaseVmax)

	# Overlay image
	cax=ax.imshow(Overlay,extent=extent,
		cmap=OverlayCmap,vmin=OverlayVmin,vmax=OverlayVmax)

	# Plot formatting
	F.colorbar(cax,orientation=cbar_orientation)
	if title:
		ax.set_title(title)


# --- Plot imagettes ---
def imagettes(imgs,mRows,nCols,cmap='viridis',downsampleFactor=0,vmin=None,vmax=None,pctmin=None,pctmax=None,colorbarOrientation=None,background=None,
	extent=None,showExtent=False,titleList=None,supTitle=None):
	# If images are in a list, convert to 3D data cube
	if len(imgs)>1:
		imgs=np.array(imgs)

	# Number of imagettes
	nImgs=imgs.shape[0]

	# Number of imagettes per figure
	nbImagettes=mRows*nCols

	# Loop through image list
	x=1 # position variable
	for i in range(nImgs):
		# Generate new figure if needed
		if x%nbImagettes==1:
			F=plt.figure() # new figure
			x=1 # reset counter

		# Format image
		img=imgs[i,:,:] # current image from list

		ds=int(2**downsampleFactor) # downsample factor
		img=img[::ds,::ds] # downsample image

		if background is not None:
			if background=='auto':
				backgroundValue=imgBackground(img)
			else:
				backgroundValue=background
			img=np.ma.array(img,mask=(img==backgroundValue))

		if pctmin is not None:
			assert vmin is None and vmax is None, 'Specify either vmin/max or pctmin/max, not both'
			stats=mapStats(img,pctmin=pctmin,pctmax=pctmax); vminValue=stats.vmin; vmaxValue=stats.vmax 
		else:
			vminValue=vmin; vmaxValue=vmax

		# Plot image as subplot
		ax=F.add_subplot(mRows,nCols,x)
		cax=ax.imshow(img,extent=extent,
			cmap=cmap,vmin=vminValue,vmax=vmaxValue)

		# Plot formatting
		if titleList:
			if type(titleList[i]) in [float,np.float64]:
				titleList[i]=round(titleList[i],2)
			ax.set_title(titleList[i])
		else:
			ax.set_title(i)

		if supTitle:
			F.suptitle(supTitle)

		if showExtent is False:
			ax.set_xticks([]); ax.set_yticks([])

		if colorbarOrientation:
			F.colorbar(cax,orientation=colorbarOrientation)

		x+=1 # update counter


#################################
### --- Date pair viewing --- ###
#################################

# --- Plot pairs ---
def plotDatePairs(pairs):
	# Provide pairs as a nested list of lists, e.g.,
	#  [[20190101,20181201]
	#   [20181201,20181101]]

	Fig=plt.figure()
	ax=Fig.add_subplot(111)
	n=0
	for pair in pairs:
		ax.plot(pair,[n,n],'-k.')
		n+=1