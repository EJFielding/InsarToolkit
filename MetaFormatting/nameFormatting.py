import os
from datetime import datetime, time

# --- Sentinel SLC name parser --- 
# Parse Sentinel SLC names 
#	Key: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions 
class SentinelName:
	def __init__(self,fpath): 
		self.name=os.path.basename(fpath) 
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
		self.name=os.path.basename(name).strip('.nc') 
		# Split into parts 
		parts=self.name.split('-')  
		self.sensor=parts[0]   # sensor name 
		self.dataset=parts[1]  # dataset name 
		self.orient=parts[2]   # satellite orientation 
		self.look=parts[3]     # sat. look direction 
		self.track=parts[4]    # track number 
		self.mode=parts[5]     # acquisition mode 
		self.pair=parts[6]     # acquisition date  
		self.time=parts[7]     # acquisition time 
		self.coords=parts[8]   # acquisition coords 
		self.orbits=parts[9]   # orbit parameters 
		self.SysTag=parts[10]  # system tag 
		self.VersTag=parts[11] # version tag 
		# Date formatting 
		self.dates=self.pair.split('_') 
		RefDate=self.dates[0] # reference date 
		self.RefDate=RefDate # full date 
		self.RefYear=int(RefDate[:4]) # YYYY 
		self.RefMo=int(RefDate[4:6])  # MM 
		self.RefDay=int(RefDate[6:8]) # DD 
		SecDate=self.dates[1] # secondary date 
		self.SecDate=SecDate # full date 
		self.SecYear=int(SecDate[:4]) # YYYY 
		self.SecMo=int(SecDate[4:6])  # MM 
		self.SecDay=int(SecDate[6:8]) # DD 
		# Time formatting 
		self.hh=int(self.time[:2])  # hours 
		self.mm=int(self.time[2:4]) # minutes 
		self.ss=int(self.time[4:6]) # seconds 
		self.time=time(hour=self.hh,minute=self.mm,second=self.ss)
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
