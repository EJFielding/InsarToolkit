#!/usr/bin/env python3 
# Build list of unique "epochs" (dates/times) for which 
#  weather corrections should be downloaded 

import numpy as np 
from datetime import datetime, timedelta 
from InsarFormatting import * 


# --- Inputs --- 
Flist_name='/Users/rzinke/Documents//Tibet/AtmoCorrection/Weather_download_names/filenames.txt' 
WeatherDate_name='/Users/rzinke/Documents//Tibet/AtmoCorrection/Weather_download_names/weather_dates.txt' 
WeatherDateUnique_name='/Users/rzinke/Documents//Tibet/AtmoCorrection/Weather_download_names/unique_weather_dates.txt' 

n=2 # number of closest times (1 <= n <= 6)

vocal=False 

# --- Setup --- 
# All dates/times of weather samples 
AllWeatherDates=[] # empty list to fill with weather dates 
Wfile=open(WeatherDate_name,'w') # output text file 
Wfile.write('# Epoch, Weather epoch\n') # write header 

# Unique-only dates/times of weather samples 
UniqueWeatherDates=[] # empty list to fill with unique dates 
UWfile=open(WeatherDateUnique_name,'w')  # output text file 
UWfile.write('# Unique weather epoch\n') # write header 

# --- List of files ---  
FlistFile=open(Flist_name,'r') # open list of filenames 
Flist=FlistFile.readlines() # read in filenames 
FlistFile.close() # close file 

## Sort through list of files 
for F in Flist: 
	# Extract date and time 
	S=ARIAname(F.strip('\n')) # format filename string 
	RefDate=datetime(S.RefYear,S.RefMo,S.RefDay,S.hh,S.mm,S.ss) 
	SecDate=datetime(S.SecYear,S.SecMo,S.SecDay,S.hh,S.mm,S.ss) 

	## --- 
	## Find closest dates/times for Reference 
	# Weather sample times 
	PrevDay=RefDate-timedelta(days=1) 
	NextDay=RefDate+timedelta(days=1) 

	Tweather=np.array([
		datetime(PrevDay.year,PrevDay.month,PrevDay.day,18,0,0), 
		datetime(RefDate.year,RefDate.month,RefDate.day,0,0,0), 
		datetime(RefDate.year,RefDate.month,RefDate.day,6,0,0), 
		datetime(RefDate.year,RefDate.month,RefDate.day,12,0,0), 
		datetime(RefDate.year,RefDate.month,RefDate.day,18,0,0),
		datetime(NextDay.year,NextDay.month,NextDay.day,0,0,0)  
		]) 
	# Differences from acquisition time 
	Tdiffs=np.zeros(6) 
	for i in range(6): 
		diff=RefDate-Tweather[i] # calculate difference 
		Tdiffs[i]=np.abs(diff.total_seconds()) # express in total seconds and record 

	ndx=np.argsort(Tdiffs) # sort from smallest-largest difference 

	# Print outputs so far 
	if vocal is True: 
		print('%s' % S.name) 
		print('RefDate',RefDate) 
		print('tweather',Tweather)
		print('time diff',Tdiffs)
		print('ndx',ndx) 

	# Record closest times 
	for i in range(n): 
		# Write to file 
		Wfile.write('%s,%s\n' % (str(RefDate),str(Tweather[ndx[i]]))) 
		# Append to list 
		AllWeatherDates.append(str(Tweather[ndx[i]])) 
		if vocal is True: 
			print('\t weather epoch',Tweather[ndx[i]]) 

	## --- 
	## Find closest dates/times for Secondary 
	# Weather sample times 
	PrevDay=SecDate-timedelta(days=1) 
	NextDay=SecDate+timedelta(days=1) 

	Tweather=np.array([
		datetime(PrevDay.year,PrevDay.month,PrevDay.day,18,0,0), 
		datetime(SecDate.year,SecDate.month,SecDate.day,0,0,0), 
		datetime(SecDate.year,SecDate.month,SecDate.day,6,0,0), 
		datetime(SecDate.year,SecDate.month,SecDate.day,12,0,0), 
		datetime(SecDate.year,SecDate.month,SecDate.day,18,0,0),
		datetime(NextDay.year,NextDay.month,NextDay.day,0,0,0)  
		]) 
	# Differences from acquisition time 
	Tdiffs=np.zeros(6) 
	for i in range(6): 
		diff=SecDate-Tweather[i] # calculate difference 
		Tdiffs[i]=np.abs(diff.total_seconds()) # express in total seconds and record 

	ndx=np.argsort(Tdiffs) # sort from smallest-largest difference 

	# Print outputs so far 
	if vocal is True: 
		print('SecDate',SecDate) 
		print('tweather',Tweather)
		print('time diff',Tdiffs)
		print('ndx',ndx) 

	# Record closest times 
	for i in range(n): 
		# Write to file 
		Wfile.write('%s,%s\n' % (str(SecDate),str(Tweather[ndx[i]]))) 
		# Append to list 
		AllWeatherDates.append(str(Tweather[ndx[i]])) 
		if vocal is True: 
			print('\t weather epoch',Tweather[ndx[i]]) 

## --- 
## Keep unique weather dates 
for d in AllWeatherDates: 
	if d not in UniqueWeatherDates: 
		# Write to file 
		UWfile.write('%s\n' % (str(d)))
		# Append to list 
		UniqueWeatherDates.append(d) 
		if vocal is True: 
			print('%s' % (str(d)))


# --- Closeout --- 
# Summary statements 
print('%i acquisition epochs' % (2*len(Flist)))
print('%i total epochs considering n = %i nearest times' % (len(AllWeatherDates),n)) 
print('%i unqiue weather epochs' % (len(UniqueWeatherDates)))

# Close files  
Wfile.close()  # close weather sample file 
UWfile.close() # close unique weather sample file 


