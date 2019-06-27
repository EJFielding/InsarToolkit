# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Format insar names and images 
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
from datetime import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 


#################################
### --- String formatting --- ###
#################################

# --- Degrees to meters --- 
def deg2m(degrees): 
	# Convert 
	m=degrees*111.111  
	return m 


# --- Sentinel SLC name parser --- 
# Parse Sentinel SLC names 
#	Key: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions 
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


# --- ARIA standard product name parser --- 
# Parse ARIA product names 
#	Key: https://aria.jpl.nasa.gov/node/97 
class ARIAname: 
	def __init__(self,name): 
		self.name=name.strip('.nc') 
		# Split into parts 
		parts=self.name.split('-')  
		self.sensor=parts[0]   # sensor name 
		self.dataset=parts[1]  # dataset name 
		self.orient=parts[2]   # satellite orientation 
		self.look=parts[3]     # sat. look direction 
		self.track=parts[4]    # track number 
		self.mode=parts[5]     # acquisition mode 
		self.date=parts[6]     # acquisition date  
		self.time=parts[7]     # acquisition time 
		self.coords=parts[8]   # acquisition coords 
		self.orbits=parts[9]   # orbit parameters 
		self.SysTag=parts[10]  # system tag 
		self.VersTag=parts[11] # version tag 
		# Date formatting 
		dates=self.date.split('_') 
		RefDate=dates[0] # reference date 
		self.RefDate=RefDate # full date 
		self.RefYear=int(RefDate[:4]) # YYYY 
		self.RefMo=int(RefDate[4:6])  # MM 
		self.RefDay=int(RefDate[6:8]) # DD 
		SecDate=dates[1] # secondary date 
		self.SecDate=SecDate # full date 
		self.SecYear=int(SecDate[:4]) # YYYY 
		self.SecMo=int(SecDate[4:6])  # MM 
		self.SecDay=int(SecDate[6:8]) # DD 
		# Time formatting 
		self.hh=int(self.time[:2])  # hours 
		self.mm=int(self.time[2:4]) # minutes 
		self.ss=int(self.time[4:6]) # seconds 
		# Coordinate formatting 
		C1=self.coords.split('_')[0] 
		self.C1='%s.%s' % (C1[:2],C1[2:5]) 
		self.hemi1=C1[-1] 
		C2=self.coords.split('_')[1] 
		self.C2='%s.%s' % (C2[:2],C2[2:5]) 
		self.hemi2=C2[-1] 
	# Print if specified 
	def vocal(self): 
		print('Name: %s' % self.name) 
		print('\tSensor [sensor]: %s' % self.sensor) 
		print('\tDataset name [dataset] (GUNW,GUNW_COSEISMIC): %s' % self.dataset) 
		print('\tSatellite orient. [AD] (A,D): %s' % self.orient) 
		print('\tLook direction [look] (L/R): %s' % self.look) 
		print('\tTrack numbert [track]: %s' % self.track) 
		print('\tAcq. mode [mode] (tops,stripmap,etc): %s' % self.mode) 
		print('\tOrbit precision [orbits] (PP,RP,PR,RR): %s' % self.orbits) 
		print('\tSystem tag [SysTag] (unique tag): %s' % self.SysTag) 
		print('\tVersion tag [VersTag]: %s' % self.VersTag) 
		print('\tReference Date [RefDate]') 
		print('\t\tRef Year:  ',self.RefYear) 
		print('\t\tRef Month: ',self.RefMo) 
		print('\t\tRef Day:   ',self.RefDay) 
		print('\tSecondary Date [SecDate]') 
		print('\t\tSec Year:  ',self.SecYear) 
		print('\t\tSec Month: ',self.SecMo) 
		print('\t\tSec Day:   ',self.SecDay) 
		print('\tCenter Time of Product [time]') 
		print('\t\tHours [hh]  ',self.hh) 
		print('\t\tMinutes [mm]',self.mm) 
		print('\t\tSeconds [ss]',self.ss) 
		print('\tCoords of west corners [coords]',self.coords) 
		print('\t\tC1 [C1,hemi1]:',self.C1,self.hemi1) 
		print('\t\tC2 [C2,hemi2]:',self.C2,self.hemi2) 


###############################
### --- Time formatting --- ###
###############################

# --- Time in seconds --- 
# Convert hours,min,sec to total seconds 
def hms2sec(h,m,s): 
	total_sec=int(h)*3600+int(m)*60+int(s) 
	return total_sec 


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


####################################
### --- Geographic transform --- ###
####################################

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
		self.bounds=[self.xmin,self.ymin,self.xmax,self.ymax] 
		# Print outputs? 
		if vocal is not False: 
			print('Image properties: ')
			print('\tNS-dim (m): %i' % self.m) 
			print('\tEW-dim (n): %i' % self.n) 
			print('\tystart: %f\tyend: %f' % (self.ystart,self.yend)) 
			print('\txstart: %f\txend: %f' % (self.xstart,self.xend)) 
			print('\tystep: %f\txstep: %f' % (self.ystep,self.xstep)) 
