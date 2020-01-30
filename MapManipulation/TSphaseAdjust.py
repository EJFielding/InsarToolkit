# Adjust phase maps to a common point or patch of pixels 

# Take a stack of interferograms and set one point in the stack to be zero 
# Find the point or patch manually or automatically 
# Output to file or memory 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 

def phaseAdjust(IFGs,location,loc_type,expc='median',vocal=False,plot=False): 
	# INPUTS 
	#	IFGs is a list of interferograms 
	#	location is where all the values will be set to zero 
	#	loc_type tells the script how to read the location
	#		'ImgCoords' [n, m, n+j, m+i] (pixels) 
	#		'MapCoords' [W, S, E, W]
	#	expc is the expectation operator 
	#		'mean', 'median' 
	# OUTPUTS 

	# Basic parameters 
	Ni=len(IFGs) 
	# Location type 
	if loc_type is 'ImgCoords': 
		n=location[0]; m=location[1] 
		n2=location[2]; m2=location[3] 
		h=location[2]-n; w=location[3]-m 
	elif loc_type is 'MapCoords': 
		print('! This functionality does not work yet!')
	# Expectation operator 
	if expc is 'mean': 
		E=np.mean 
	elif expc is 'median': 
		E=np.median 

	# Initial printouts 
	if vocal is True: 
		print('Number interferograms: %i' % (Ni)) 
	if plot is True: 
		nSubPlots=np.ceil(np.sqrt(Ni)) # rows/cols of subplots 
		Fo=plt.figure() # set up original plot 
		Fa=plt.figure() # set up adjusted plot 

	# Adjust phase maps 
	for i in range(Ni): 
		# Sample patch 
		Svals=IFGs[i].P[m:m2,n:n2] # sample values 
		Svals=np.ma.array(Svals,mask=(Svals==IFGs[i].BG)) # mask no data 
		IFGs[i].SampPatch=E(Svals.compressed()) # expectation 
		# Possible outputs before patch removal 
		if vocal is True: 
			print('%s value: %f' % (IFGs[i].name,IFGs[i].SampPatch))
		if plot is True: 
			# Phase map 
			ax=Fo.add_subplot(nSubPlots,nSubPlots,i+1) 
			P=np.ma.array(IFGs[i].P,mask=(IFGs[i].P==IFGs[i].BG)) 
			vmin,vmax=np.percentile(P,(1,99))
			cax=ax.imshow(P,vmin=vmin,vmax=vmax) 
			ax.set_yticks([]); ax.set_yticklabels([]) 
			ax.set_xticks([]); ax.set_xticklabels([]) 
			Fo.colorbar(cax,orientation='horizontal') 
			# Sample patch 
			ax.add_patch(Rectangle((n,m),w,h,color=(0.6,0.55,0.5))) 
		# Remove sample patch values from interferograms 
		IFGs[i].P-=IFGs[i].SampPatch 
		# Possible outputs after patch removal 
		if vocal is True: 
			print('\tmean residuals: %f' % np.mean(np.abs(IFGs[i].P[m:m2,n:n2]))) 
			print('\tmedian residuals: %f' % np.median(Svals-IFGs[i].SampPatch)) 
		if plot is True: 
			ax=Fa.add_subplot(nSubPlots,nSubPlots,i+1) 
			vmin,vmax=np.percentile(IFGs[i].P,(1,99)) 
			cax=ax.imshow(IFGs[i].P,vmin=vmin,vmax=vmax) 
			ax.set_yticks([]); ax.set_yticklabels([]) 
			ax.set_xticks([]); ax.set_xticklabels([]) 
			Fa.colorbar(cax,orientation='horizontal') 
			# Sample patch 
			ax.add_patch(Rectangle((n,m),w,h,color=(0.6,0.55,0.5),fill=False)) 