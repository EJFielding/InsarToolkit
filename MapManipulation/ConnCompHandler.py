# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Handle and manipulate connected components files  
# 
# Functions 
# cmpStats - provides statistics of conn comp maps 
# cmpOrder - reorders conn comp based on percent of total extent 
# cmpIsolate - provides maps of single conn comps 
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 


####################################
### --- CONNECTED COMPONENTS --- ###
####################################

# --- Component statistics --- 
# Statistical measures of connected components 
class cmpStats: 
	def __init__(self,I,vocal=False,hist=False): 
		m,n=I.shape # shape of array 
		N=m*n # total data points 
		# Find unique values 
		self.unique=np.unique(I) 
		self.nUnique=len(self.unique) 
		# Counts for each unique value 
		self.uCount=np.zeros(self.nUnique) # empty array for count values 
		self.pct=np.zeros(self.nUnique) # empty array for percents of total 
		for i in range(self.nUnique): 
			boolArray=(I==self.unique[i]) # True/False array 
			self.uCount[i]=np.sum(boolArray) # update array with counts 
			self.pct[i]=self.uCount[i]/N # percent of total 
		if vocal is True: 
			print('Unique:',self.unique) 
			print('Counts:',self.uCount) 
			print('Pcts:  ',self.pct) 
		if hist is True: 
			nBins=int(self.unique.max()-self.unique.min()+1) 
			H,Hedges=np.histogram(I,nBins) 
			Hcntrs=np.round(Hedges[:-1]+np.diff(Hedges)/2) 
			Fhist=plt.figure(); ax=Fhist.add_subplot(111) 
			ax.bar(Hcntrs,H,color='b') 
			ax.set_xticks(Hcntrs) 
			ax.set_xlabel('component'); ax.set_ylabel('occurrence') 


# --- Order connected components --- 
# Re-name components by frequency of occurrence 
def cmpOrder(I,BG='auto',vocal=False,hist=False): 
	# INPUTS 
	#	I is the mxn image file 
	# OUTPUTS 
	#	Isort is the modified connected components file 

	# Setup --- 
	Isort=I.copy() 
	mrows,ncols=I.shape # shape 
	# # Reshape image array 
	# I=I.reshape(1,-1).squeeze(0) 
	# Background value --- 
	if BG is 'auto': 
		# Estimate background value by averaging the values along all four edges 
		top=np.mean(I[0,:]); bottom=np.mean(I[-1,:])  
		left=np.mean(I[:,0]); right=np.mean(I[:,-1]) 
		if vocal is True: 
			print('Estimating background (ave values): ') 
			print('\ttop: %.1f\n\tbottom: %.1f\n\tleft: %.1f\n\tright: %.1f' % \
				(top,bottom,left,right)) 
		# Check robustness (absolute difference between sides)
		if np.sum(np.abs(np.diff([top,bottom,left,right])))>(1E-6):
			print('! WARNING ! Sides yield different values. Background value might not be robust.') 
		# Background = average of sides 
		BG=np.round(np.mean([top,bottom,left,right])) 
	else:
		# In case of user specification 
		BG=BG 
	if vocal is True: 
		print('Initial background value: %i' % (BG)) 
	# Editing --- 
	# Replace invalid values 
	if np.sum(I[I<0])<0: 
		# Replace neg. values 
		I[I<0]=BG # set to background 
		if vocal is True:
			print('! Negative values replaced by BG value: %i' % (BG)) 
	# Compute preliminary statistics 
	C=cmpStats(I,vocal=vocal) 
	# Reorder --- 
	#	Re-name components by frequency of occurrence 
	#	(Background value is always component 0) 
	C.uCount[C.unique==BG]=-1 
	if vocal is True: 
		print('Ordering values by occurrence') 
		print('... Set background value occurrence to -1')  
	ndxSort=np.argsort(C.uCount) # sort by occurrence  
	ndxSort=ndxSort[::-1]  # list of indices 
	C.unique=C.unique[ndxSort] # sort most frequent to least 
	C.unique=C.unique[:-1] # leave off background value for now 
	if vocal is True: 
		print('Unique order:',C.unique) 
	# Assign values to sorted array 
	for i in range(C.nUnique-1): 
		Isort[I==C.unique[i]]=int(i+1) # component value 
	Isort[I==BG]=0 # handle background value 
	# Compute final statistics 
	FC=cmpStats(Isort,vocal=vocal,hist=hist) 
	# Return sorted connected components 
	return Isort 


# --- Isolate connected components --- 
# Show the phase map associated with each connected component 
def cmpIsolate(Phs,Cmp,cmpList=[1],NoData_out='mask',vocal=False,plot=False): 
	# INPUTS 
	#	Phs is a 2D array that is the phase map 
	#	Cmp is a 2D array that is the conn comp map 
	#	cmpList is a list with the desired components 
	#	NoData_out is a float value or NaN, if 'mask' values will be retained 
	# OUTPUTS 
	#	PhsMap is a dictionary with each phase map encoded 
	#	 as an entry 
	#	CmpMap is similar to PhsMap 
	nC=len(cmpList) # number of components 
	assert Phs.shape==Cmp.shape # check same size 
	q,r=Phs.shape # orig image dimensions 
	PhsMap={}; CmpMap={} # empty dictionaries 
	for c in cmpList: 
		if vocal is True: 
			print('\tcomponent: %i' % c) 
		if NoData_out=='mask':
			PhsMap[c]=np.ma.array(Phs,mask=(Cmp!=c)) 
			CmpMap[c]=np.ma.array(Cmp,mask=(Cmp!=c)) 
		else: 
			PhsMap[c]=Phs.copy() # add values 
			PhsMap[c][Cmp!=c]=NoData_out # null nodata values 
			CmpMap[c]=Cmp.copy() # add values 
			CmpMap[c][Cmp!=c]=NoData_out # null nodata values 
		if plot is True: 
			F=plt.figure('Comp%i' % c) 
			axP=F.add_subplot(1,2,1) 
			caxP=axP.imshow(PhsMap[c]) 
			axP.set_aspect(1); axP.set_title('Phase') 
			F.colorbar(caxP,orientation='horizontal') 
			axC=F.add_subplot(1,2,2) 
			caxC=axC.imshow(CmpMap[c]) 
			axC.set_aspect(1); axC.set_title('Comp') 
			F.colorbar(caxC,orientation='horizontal') 
	return PhsMap, CmpMap 



