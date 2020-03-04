# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic, general mathematical functions
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
from datetime import datetime, time
import numpy as np 


###############################
### --- Time formatting --- ###
###############################

# --- Unique dates from pairs ---
# Find unique dates given a list of date pairs
def udatesFromPairs(datePairs,verbose=False):
	allDates=[] # empty list of all dates
	[allDates.extend(pair) for pair in datePairs]
	nAllDates=len(allDates)

	uniqueDates=[] # empty list for unique dates
	[uniqueDates.append(d) for d in allDates if d not in uniqueDates]
	nUniqueDates=len(uniqueDates)

	if verbose is True:
		print('Individual dates: {}'.format(nAllDates))
		print('Unique dates: {}'.format(nUniqueDates))

	return uniqueDates


# --- Time in seconds --- 
# Convert hours,min,sec to total seconds 
def hms2sec(h,m,s): 
	total_sec=int(h)*3600+int(m)*60+int(s) 
	return total_sec 


# --- Time in hms ---
# Convert total seconds to hours,min,sec
def sec2hms(total_sec):
	h=int(total_sec//3600) # hours
	total_sec%=3600 # update time left over
	m=int(total_sec//60) # minutes
	total_sec%=60 # update time left over
	s=int(total_sec) # seconds
	return h,m,s


# --- SAR date ---
# Format YYYYMMDD:hhmmss into Python date format
class SARdate: 
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


#################################
### --- Date calculations --- ###
#################################

# --- Date difference ---
# Simpler and more compact than "dateDiff"
#  function below
def daysBetween(d1,d2,fmt="%Y%m%d",absTime=True):
	d1 = datetime.strptime(str(d1),fmt)
	d2 = datetime.strptime(str(d2),fmt)
	days=(d2-d1).days
	if absTime is True:
		days=np.abs(days)
	return days


# --- Date difference --- 
# Difference two dates in format YYYYMMDD:hhmmss to 
#	retrieve a single value of time with 
#	desired units (e.g., years; seconds) 
def dateDiff(date1,date2,fmt='yr',vocal=False): 
	## Check input formats 
	#	YYYYMMDD is 8 char long 
	#	YYYYMMDD_hhmmss is 17 char long 
	n1=len(date1)
	assert n1==8 or n1 in range(14,16), "Check date1 format length."  
	n2=len(date2) 
	assert n2==8 or n2 in range(14,16), "Check date2 format length." 

	## Format into Python dates
	# Parse img 1 into age units 
	D1=SARdate('date1',date1) # create instance 
	# Parse img 2 into age units 
	D2=SARdate('date2',date2) # create instance 
	# Convert YYYYMMDD to useable date 
	D1.date=datetime(D1.yr,D1.mo,D1.dy,D1.hr,D1.mn,D1.sc) 
	D2.date=datetime(D2.yr,D2.mo,D2.dy,D2.hr,D2.mn,D2.sc) 

	## Difference dates
	# Calculate time difference in seconds  
	dateDiff=D2.date-D1.date 
	tDiff=dateDiff.days*24*60*60 # day*hr*min*sec 
	tDiff+=dateDiff.seconds # hr*min*sec 
	# Convert seconds to desired output 
	if fmt in ['y', 'yr', 'year', 'years']: 
		fmt='years' 
		tDiffFmt=tDiff/31536000 # sec/yr 
	elif fmt in ['m', 'mo', 'month', 'months']:
		fmt='ave months' 
		tDiffFmt=tDiff/2628002.88 # sec/mo 
	elif fmt in ['d', 'dy', 'day', 'days']: 
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


# --- Average time ---
# Compute the average time given a list of times
class avgTime:
	def __init__(self,TimeList,vocal=False):
		# Check formatting is appropriate
		if isinstance(TimeList[0],time):
			print('Reading as datetime.time object')
		elif hasattr(TimeList[0],'hour') and hasattr(TimeList[0],'minute') and hasattr(TimeList[0],'second'):
			print('Reading as custom time object')
		assert hasattr(TimeList[0],'hour') and hasattr(TimeList[0],'minute') and hasattr(TimeList[0],'second'), \
		'Please format as datetime.time or similar object'

		# Find average in seconds
		nTimes=len(TimeList) # number of time recordings in list
		Seconds=0 # convert list of times into number of seconds
		for t in TimeList:
			s=hms2sec(t.hour,t.minute,t.second) # convert time to seconds
			Seconds+=s # keep running list of cumulative seconds
		Seconds/=nTimes # divide for average
		if vocal is True:
			print('Number of time samples: {}'.format(nTimes))
			print('Avergage time (seconds): {}'.format(Seconds) )

		# Convert seconds back to hours, minutes, seconds
		self.total_seconds=Seconds # store to object
		self.h,self.m,self.s=sec2hms(Seconds) # hms format
		self.time=time(hour=self.h,minute=self.m,second=self.s)
		if vocal is True:
			print('Average time: {}:{}:{}'.format(self.h,self.m,self.s))


# --- Cumulative time ---
def cumulativeTime(datePairs,absTime=True):
	nPairs=len(datePairs) # number of pairs

	cumulative_time=np.zeros(nPairs)
	cumulative_time[0]=daysBetween(datePairs[0][0],datePairs[0][1],absTime=absTime)
	for i in range(1,nPairs):
		pair=datePairs[i]
		cumulative_time[i]=cumulative_time[i-1]+daysBetween(pair[0],pair[1])

	# Normalize to years
	cumulative_time/=365.25

	return cumulative_time


########################
### --- Triplets --- ###
########################

# Triplets
def createTriplets(dates,lags=1,minTime=None,maxTime=None,verbose=False):
	"""
		Provide a list of unique dates in format YYYYMMDD. This
		 function will create a list of the (n1-n0, n2-n1, n2-n0)
		 phase triplets. 
		It does not accept a list of pairs like the "formatTriplets"
		 function below.

		Lags is the minimum interval from one acquisition to the 
		 next. For instance:
			lags=1 gives [n1-n0, n2-n1, n0-n2]
			lags=2 gives [n2-n0, n4-n2, n0-n4]
			lags=3 gives [n3-n0, n6-n3, n0-n6]
	"""

	# Loop through dates to create valid triplet combinations
	nDates=len(dates)
	triplets=[]
	for n in range(nDates-2*lags):
		dateI=dates[n] # first date in sequence
		dateJ=dates[n+lags] # second date in sequence
		dateK=dates[n+2*lags] # third date in sequence
		pairList=[[dateI,dateJ],[dateJ,dateK],[dateI,dateK]]
		triplets.append(pairList) # add to list

	# Check that pairs meet time requirements
	if minTime:
		# Convert pairs to intervals in days
		intervals=[]
		for triplet in triplets:
			intervalSet=[daysBetween(pair[0],pair[1]) for pair in triplet]
			intervals.append(min(intervalSet))
		validTriplets=[triplet for ndx,triplet in enumerate(triplets) if intervals[ndx]>=int(minTime)]
		triplets=validTriplets

	if maxTime:
		# Convert pairs to intervals in days
		intervals=[]
		for triplet in triplets:
			intervalSet=[daysBetween(pair[0],pair[1]) for pair in triplet]
			intervals.append(max(intervalSet))
		print(intervals)
		validTriplets=[triplet for ndx,triplet in enumerate(triplets) if intervals[ndx]<=int(maxTime)]
		triplets=validTriplets

	# Print if requested
	if verbose is True:
		print('Triplets...')
		print('{} unique dates for triplet formulation'.format(nDates))
		print('Triplets:'); [print(triplet) for triplet in triplets]
		print('{} triplets created'.format(len(triplets)))

	return triplets



# Triplets from list of pairs
def formatTriplets(datePairs,minTime=None,maxTime=None,verbose=False):
	"""
		Provide a list of date pairs to retrieve a list of phase
		 triplets. 
	"""

	# Loop through date pairs to find all valid triplet combinations
	nDatePairs=datePairs.shape[0] # number of date pairs
	triplets=[]
	for pairIJ in datePairs:
		# Pair IJ - first date; second date
		dateI=pairIJ[0]; dateJ=pairIJ[1]
		# Pairs JK - pairs with second date as master
		pairsJK=datePairs[datePairs[:,0]==dateJ]
		if len(pairsJK)>0:
			# Pairs IK - pairs with I as master and K as slave
			for dateK in pairsJK[:,1]:
				pairsIK=datePairs[(datePairs[:,0]==dateI) & (datePairs[:,1]==dateK)]
				if len(pairsIK)>0:
					# Record valid date pairs to list
					pairList=[[dateI,dateJ],[dateJ,dateK],[dateI,dateK]]
					triplets.append(pairList)
	nTriplets=len(triplets)

	# Remove pairs that do not meet temporal requirements
	failed_conditions=[] # make a list of indexes where conditions not met
	for t in range(nTriplets):
		# List pairs
		pairIJ=triplets[t][0]
		pairJK=triplets[t][1]
		pairIK=triplets[t][2]

		# Compute time intervals
		IJdt=days_between(pairIJ[0],pairIJ[1])/365
		JKdt=days_between(pairJK[0],pairJK[1])/365
		IKdt=days_between(pairIK[0],pairIK[1])/365

		# Check against conditions
		if maxTime: # maxTime
			conditions_met=[False,False,False]
			conditions_met[0]=True if IJdt<maxTime else False
			conditions_met[1]=True if JKdt<maxTime else False
			conditions_met[2]=True if IKdt<maxTime else False
			# Record index if not all maxTime conditions met
			if sum(conditions_met)<3:
				failed_conditions.append(t)

		if minTime: # minTime
			conditions_met=[False,False,False]
			conditions_met[0]=True if IJdt>minTime else False
			conditions_met[1]=True if JKdt>minTime else False
			conditions_met[2]=True if IKdt>minTime else False
			# Record index if not all minTime conditions met
			if sum(conditions_met)<3:
				failed_conditions.append(t)

	# Unique values of failed conditions
	failed_conditions=list(set(failed_conditions))
	failed_conditions=failed_conditions[::-1] # work from back-forward
	[triplets.pop(c) for c in failed_conditions]

	# Report if requested
	if verbose is True:
		print('Triplets:')
		[print(triplet) for triplet in triplets]
		print(nTriplets)
	return triplets


############################
### --- For HDF DATA --- ###
############################

# --- Format date list ---
def formatHDFdates(dateDS,verbose=False):
	# List of master-slave date pairs
	datePairs=dateDS[:,:].astype('int') # Reformat as appropriate data type
	# List of unique dates
	allDates=[]; [allDates.extend(pair) for pair in datePairs] # add dates from pairs
	dates=[]; [dates.append(d) for d in allDates if d not in dates] # limit to unique dates
	if verbose is True:
		print('HDF5 date pairs:\n{}'.format(datePairs))
		print('HDF5 unique dates:\n{}'.format(dates))
	return dates, datePairs


#############################
### --- For GDAL DATA --- ###
#############################
