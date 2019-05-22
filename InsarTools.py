# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specific tools for interpreting/processing
#	InSAR products 
# By Rob Zinke, 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
from datetime import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 


##########################
### --- Formatting --- ###
##########################

# --- Degrees to meters --- 
def deg2m(degrees): 
	# Convert 
	m=degrees*111.111  
	return m 

# --- Image name parser --- 
# Parse Sentinel image name 
class SentinelName:
	def __init__(self,fpath): 
		self.name=fpath.split('/')[-1] 
	def parse(self): 
		fname=self.name.replace('__','_') 
		name_parts=fname.split('_') 
		self.mission=name_parts[0] # satellite platform, e.g., S1B
		self.mode=name_parts[1] # mode/beam 
		self.product=name_parts[2] # product type, e.g., RAW; SLC 
		self.misc=name_parts[3] # misc parameters 
		self.proc_lvl=self.misc[0] # processing level, e.g., 1, 2
		self.prod_class=self.misc[1] # product class 
		self.polarization=self.misc[2:] # polarization 
		# Times of Acquisition 
		self.AcqStart=name_parts[4] # full date of acquisition 
		self.AcqDateStart=self.AcqStart.split('T')[0] 
		self.AcqYearStart=self.AcqStart[0:4] 
		self.AcqMonthStart=self.AcqStart[4:6] 
		self.AcqDayStart=self.AcqStart[6:8] 
		self.AcqTimeStart=self.AcqStart.split('T')[1] 
		self.AcqHHstart=self.AcqTimeStart[0:2] # hours 
		self.AcqMMstart=self.AcqTimeStart[2:4] # minutes 
		self.AcqSSstart=self.AcqTimeStart[4:6] # seconds 
		self.AcqEnd=name_parts[5] # full end of acquisition 
		self.AcqDateEnd=self.AcqEnd.split('T')[0] 
		self.AcqYearEnd=self.AcqEnd[0:4] 
		self.AcqMonthEnd=self.AcqEnd[4:6] 
		self.AcqDayEnd=self.AcqEnd[6:8] 
		self.AcqTimeEnd=self.AcqEnd.split('T')[1] 
		self.AcqHHend=self.AcqTimeEnd[0:2] # hours 
		self.AcqMMend=self.AcqTimeEnd[2:4] # minutes 
		self.AcqSSend=self.AcqTimeEnd[4:6] # seconds 	
		# Flight Parameters 
		self.AbsOrbitNb=name_parts[6] # absolute orbit number 
		self.MissionID=name_parts[7] # Mission Data Take ID 
		self.ProdUnique=name_parts[8] # Product Unique ID 


# --- Date difference --- 
# Difference two dates in format YYYYMMDD:hhmmss to 
#	retrieve a single value of time with 
#	desired units (e.g., years; seconds) 
def dateDiff(date1,date2,fmt='yr',vocal=False): 
	# Check input formats 
	#	YYYYMMDD is 8 char long 
	#	YYYYMMDD_hhmmss is 17 char long 
	n1=len(date1)
	assert n1==8 or n1 in range(14,16), "Check date1 format length."  
	n2=len(date2) 
	assert n2==8 or n2 in range(14,16), "Check date2 format length." 
	# Define object 
	class D: 
		def __init__(self,name,date): 
			# Object attributes 
			self.name=name 
			self.n=len(date) # nb char 
			assert int(date[0:4]) in range(1990,2100), "Check %s year format." % (name) 
			self.yr=int(date[0:4]) # year 
			assert int(date[4:6]) in range(1,13), "Check %s month format." % (name) 
			self.mo=int(date[4:6]) # month 
			assert int(date[6:8]) in range(1,32), "Check %s day format." % (name) 
			self.dy=int(date[6:8]) # day 
			# Assign hours, minutes, seconds 
			if self.n>8: 
				time=date[8:] # second half of string 
				time=time.strip('T'); time=time.strip(':') # fmt 
				self.hr=int(time[0:2]) # hours 
				self.mn=int(time[2:4]) # minutes 
				self.sc=int(time[4:6]) # seconds  
			else: 
				self.hr=00 # hours 
				self.mn=00 # minutes 
				self.sc=00 # seconds 
	# Parse img 1 into age units 
	D1=D('date1',date1) # create instance 
	# Parse img 2 into age units 
	D2=D('date2',date2) # create instance 
	# Convert YYYYMMDD to useable date 
	D1.date=datetime(D1.yr,D1.mo,D1.dy,D1.hr,D1.mn,D1.sc) 
	D2.date=datetime(D2.yr,D2.mo,D2.dy,D2.hr,D2.mn,D2.sc) 
	# Calculate time difference in seconds  
	dateDiff=D2.date-D1.date 
	tDiff=dateDiff.days*24*60*60 # day*hr*min*sec 
	tDiff+=dateDiff.seconds # hr*min*sec 
	# Convert seconds to desired output 
	if fmt is 'y' or fmt is 'yr' or fmt is 'year' or fmt is 'years': 
		fmt='years' 
		tDiffFmt=tDiff/31536000 # sec/yr 
	elif fmt is 'm' or fmt is 'mo' or fmt is 'month' or fmt is 'months':
		fmt='ave months' 
		tDiffFmt=tDiff/2628002.88 # sec/mo 
	elif fmt is 'd' or fmt is 'dy' or fmt is 'day' or fmt is 'days': 
		fmt='days' 
		tDiffFmt=tDiff/86400 # sec/day 
	else: 
		fmt='seconds' 
		tDiffFmt=tDiff 
	# Outputs 
	if vocal is True: 
		print('\tDate2: %s;\tDate1: %s' % (D2.date,D1.date)) 
		print('\tTime diff: %i sec => %.6f %s' % (tDiff,tDiffFmt,fmt)) 
	return tDiffFmt 


# --- GDAL geographic transform --- 
# Format transform data into something useful 
class GDALtransform:
	def __init__(self,DS=None,transform=None,shape=None,vocal=False):
		# transform comes from data.GetGeoTransform() 
		# shape comes from data.GetRasterBand(#).ReadAsArray().shape 
		if DS is not None:
			transform=DS.GetGeoTransform() 
			shape=(DS.RasterYSize,DS.RasterXSize) 
		self.m=shape[0] 
		self.n=shape[1] 
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
		# Print outputs? 
		if vocal is not False: 
			print('Image properties: ')
			print('\tNS-dim (m): %i' % self.m) 
			print('\tEW-dim (n): %i' % self.n) 
			print('\tystart: %f\tyend: %f' % (self.ystart,self.yend)) 
			print('\txstart: %f\txend: %f' % (self.xstart,self.xend)) 
			print('\tystep: %f\txstep: %f' % (self.ystep,self.xstep)) 


###############################
### --- IMAGE FUNCTIONS --- ### 
###############################

# --- Multilooking --- 
def multiLook(infile,outfile,fmt='GTiff',xlooks=19,ylooks=7,noData=None,resampAlg='average'): 
	'''
	infile - Input file to multilook 
	outfile - Output multilooked file 
	fmt - Output format 
	xlooks - Number of looks in x/range direction 
	ylooks - Number of looks in y/azimuth direction 
	''' 
	IMG=gdal.Open(infile,gdal.GA_ReadOnly) 
	# Input dimensions 
	xSize=IMG.RasterXSize 
	ySize=IMG.RasterYSize 
	# Output dimensions 
	outXSize=xSize//xlooks 
	outYSize=ySize//ylooks 
	# Translation options 
	transOpt=gdal.TranslateOptions(format=fmt,
		width=outXSize,height=outYSize,
		srcWin=[0,0,outXSize*xlooks,outYSize*ylooks],
		noData=noData,resampleAlg=resampAlg) 
	# Translate 
	R=gdal.Translate(outfile,IMG,options=transOpt) 
	if fmt=='MEM':
		return R 


####################################
### --- CONNECTED COMPONENTS --- ###
####################################

# --- Connected components basic statistics --- 
def cmpStats(I,vocal=False,hist=False): 
	class cStats: 
		def __init__(self,I,vocal): 
			# Find unique values
			self.unique=np.unique(I) 
			self.nUnique=len(self.unique) 
			# Counts for each unique value 
			self.uCount=np.zeros(self.nUnique) # empty array 
			for i in range(self.nUnique): 
				boolArray=(I==self.unique[i]) # True/False array 
				self.uCount[i]=np.sum(boolArray) # count True's 
			if vocal is True:
				print('Unique:',self.unique) 
				print('Counts:',self.uCount) 
			if hist is True: 
				nBins=int(self.unique.max()-self.unique.min()+1) 
				H,Hedges=np.histogram(I,nBins) 
				Hcntrs=np.round(Hedges[:-1]+np.diff(Hedges)/2) 
				Fhist=plt.figure() 
				ax=Fhist.add_subplot(111) 
				ax.bar(Hcntrs,H,color='b') 
				ax.set_xticks(Hcntrs) 
				ax.set_xlabel('value'); ax.set_ylabel('occurrence')
	# Convert to 1D array 
	I=I.reshape(1,-1).squeeze(0) 
	S=cStats(I,vocal) 

# Handle and manipulate connected components 
#	Esp. re-name components by frequency of occurrence 
def cmpOrder(Img,backgroundValue='auto',vocal=False): 
	# INPUTS 
	#	I is the mxn image file 
	# OUTPUTS 
	#	C is the modified connected components file 

	# Setup --- 
	I=Img.copy() 
	mrows,ncols=I.shape # shape 
	# Background value --- 
	if backgroundValue is 'auto': 
		# Estimate background value by averaging the values along all four edges 
		top=np.sum(I[0,:])/ncols 
		bottom=np.sum(I[-1,:])/ncols 
		left=np.sum(I[:,0])/mrows 
		right=np.sum(I[:,-1])/mrows 
		if vocal is True: 
			print('Estimating background (ave values): ') 
			print('\ttop: %.1f\n\tbottom: %.1f\n\tleft: %.1f\n\tright: %.1f' % \
				(top,bottom,left,right)) 
		# Check robustness (absolute difference between sides)
		if np.sum(np.abs(np.diff([top,bottom,left,right])))>(1E-6):
			print('! WARNING ! Sides yield different values. Background value might not be robust.') 
			print('\t consider setting backgroundValue = %i' % np.round(np.mean([top,bottom,left,right]))) 
		# Background = average of sides 
		BG=np.round(np.mean([top,bottom,left,right])) 
	else:
		# In case of user specification 
		BG=backgroundValue 
	if vocal is True: 
		print('Background value set to %i' % (BG)) 
	# Reshape image array 
	I=I.reshape(1,-1).squeeze(0) 
	# Editing --- 
	# Replace invalid values 
	if np.sum(I[I<0])<0: 
		# Replace neg. values 
		I[I<0]=BG # set to background 
		if vocal is True:
			print('! Negative values replaced by BG value: %i' % (BG)) 
	# Compute preliminary statistics 
	unique=np.unique(I) # unique values 
	nUnique=len(unique) # number unique values 
	uCount=np.zeros(nUnique) # counts for each unique value 
	for i in range(nUnique): 
		boolArray=(I==unique[i]) # True/False array 
		uCount[i]=np.sum(boolArray) # sum True for each value 
	if vocal is True: 
		print('Basic statistics') 
		print('N unique vals: %i' % (nUnique)) 
		print('Value:',unique) 
		print('Count:',uCount/len(I))   
	# Re-name components by frequency of occurrence 
	#	(Background value is always component 0) 
	uCount[unique==BG]=-1 
	if vocal is True: 
		print('Ordering values by occurrence') 
		print('... Set background value occurrence to -1')  
	ndxSort=np.argsort(uCount); ndxSort=ndxSort[::-1] 
	unique=unique[ndxSort] # sort most frequent to least 
	unique=unique[:-1] # leave off BG value 
	if vocal is True: 
		print('Unique order:',unique) 
	# Use constant value shift to avoid confusion 
	startVal=unique.max() # where conncomp index leaves off 
	newVal=startVal+1 # start new conncomp index 
	for u in unique: 
		I[I==u]=newVal # change old conncomp value to new
		newVal=newVal+1 # update new conncomp index 
	I[I>startVal]-=startVal # remove constant value shift 
	I[I==BG]=0 # replace background value with standard 0 
	I=I.reshape(mrows,ncols) # back to map form 
	return I


# Show the phase map associated with each connected component 
def cmpIsolate(Phs,Cmp,cmpList=[1],NoData='mask',vocal=False,plot=False): 
	# INPUTS 
	#	Phs is a 2D array that is the phase map 
	#	Cmp is a 2D array that is the conn comp map 
	#	cmpList is a list with the desired components 
	#	NoData is a float value or NaN, if 'mask' values will be retained 
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
		if NoData is 'mask' or NoData is 'Mask':
			PhsMap[c]=np.ma.array(Phs,mask=(Cmp!=c)) 
			CmpMap[c]=np.ma.array(Cmp,mask=(Cmp!=c)) 
		else: 
			PhsMap[c]=Phs.copy() # add values 
			PhsMap[c][Cmp!=c]=NoData # null nodata values 
			CmpMap[c]=Cmp.copy() # add values 
			CmpMap[c][Cmp!=c]=NoData # null nodata values 
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


#########################
### --- STITCHING --- ###
#########################




