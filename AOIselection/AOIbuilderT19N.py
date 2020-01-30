#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from KMLwriter import Polygon2KML

# --- User inputs ---
# For a given date, plot the frame and track corners
headpath='/Users/rzinke/Documents/Tibet/ARIA_track_selection/Track19/Dates/'

CSVname='asf-2019-06-11_Des_2014-2019_Track19.csv'
DateFileName='Track19_Northern_Dates.txt'

AOIname='T19_Northern'

maxLat=45 # northern center of frame
minLat=38 # southern center of frame

maxFrames=None

print('WARNING: Requires modification of "Beam Mode Description" in CSV')
print('WARNING: Midnight crossing will likely be a problem')


# --- Check subfolders exist ---
# Fig output name
figpath='%s/Figs_%s/' % (headpath,AOIname)
if not os.path.exists(figpath):
	os.mkdir(figpath)

kmlpath='%s/KMLs_%s/' % (headpath,AOIname)
if not os.path.exists(kmlpath):
	os.mkdir(kmlpath)

# --- Open list of selected dates ---
Fdates=open(headpath+DateFileName,'r')
selectDates=Fdates.readlines()
selectDates=[d.strip('\n') for d in selectDates]
Fdates.close()

# --- Handle csv ---
# Read csv
CSV=open(headpath+CSVname,'r')
CSVlines=CSV.readlines()
CSV.close()

# Interpret csv
CSVhdr=CSVlines[0].split(',') # header
CSVinfo=CSVlines[1:] # data
if maxFrames is not None:
	CSVinfo=CSVinfo[:maxFrames] # limit to max nb frames

CornerNames=['Acquisition Date',
	'Center Lat',
	'Center Lon',
	'Near Start Lat',
	'Near Start Lon',
	'Far Start Lat',
	'Far Start Lon',
	'Near End Lat',
	'Near End Lon',
	'Far End Lat',
	'Far End Lon']

print('CSVhdr',CSVhdr)

CSVkey={}
print('Column key')
for c in CornerNames:
	i=CSVhdr.index(c)
	CSVkey[c]=i
	print('\t%s: pos %i' % (c,i))

# Structures for handling data
frames={} # dictionary for storing frame instances
class Frame:
	def __init__(self,granule):
		self.granule=granule
nFramesTotal=len(CSVinfo) # nb of all frames in csv
granuleList=[] # running list of all frames within lat bounds
frameDates=[] # keep track of unique dates
prevDate=0 # start keeping track

# Pull metadata for each frame
for csvLine in CSVinfo:
	# Recover essential data
	data=csvLine.split(',') # frame metadata
	granule=data[0] # frame name
	# Check within lat bounds
	CenterLat=float(data[CSVkey['Center Lat']])
	if (CenterLat>=minLat) and (CenterLat<=maxLat):
		ValidLat=True
	else:
		ValidLat=False
	# Check is selected date
	AcqDate=data[CSVkey['Acquisition Date']] # e.g., 2015-12-08T00:11:30.000000
	AcqDate=AcqDate.split('T')[0] # date only, e.g., 2015-12-08
	AcqDate=AcqDate.replace('-','') # remove dashes, e.g., 20151208
	if AcqDate in selectDates:
		ValidDate=True
	else:
		ValidDate=False
	# Instantiate if criteria met
	if ValidLat is True and ValidDate is True:
		granuleList.append(granule) # keep track of frame names
		frames[granule]=Frame(granule) # instantiate frame
		# Gather spatial data
		frames[granule].CenterLat=CenterLat
		CenterLon=float(data[CSVkey['Center Lon']])
		NearStartLat=float(data[CSVkey['Near Start Lat']])
		NearStartLon=float(data[CSVkey['Near Start Lon']])
		FarStartLat=float(data[CSVkey['Far Start Lat']])
		FarStartLon=float(data[CSVkey['Far Start Lon']])
		NearEndLat=float(data[CSVkey['Near End Lat']])
		NearEndLon=float(data[CSVkey['Near End Lon']])
		FarEndLat=float(data[CSVkey['Far End Lat']])
		FarEndLon=float(data[CSVkey['Far End Lon']])
		# Form polygon
		FrameCorners=Polygon([
			(NearStartLon,NearStartLat),
			(FarStartLon,FarStartLat),
			(FarEndLon,FarEndLat),
			(NearEndLon,NearEndLat)])
		frames[granule].Corners=FrameCorners # corners of each frame
		frameX,frameY=FrameCorners.exterior.xy # temporarily store corners
		# Record date - this will need to change for dateline issues
		frames[granule].AcqDate=AcqDate
		frameDates.append(AcqDate)
		# Plot according to date
		if AcqDate==prevDate: # if date is the same...
			# Add to current polygon if same date
			Track=Track.union(FrameCorners)
			# Add to current plot if same date
			ax1.fill(frameX,frameY,color=(0.6,0.6,0.6),alpha=0.5)
		else: # if date is different...
			try:
				# --- Save old data at start of each new date ---
				# Save track to kml
				TrackName='AOI-%s_%s' % (AOIname,prevDate)
				T=Polygon2KML(name=TrackName,polygon=Track)
				T.write(outFilename='%sAOI-%s_%s.kml' % \
					(kmlpath,AOIname,prevDate))
				# Save previous figure if exists
				trackX,trackY=Track.exterior.xy # final track corners
				ax2.fill(trackX,trackY,color='b',alpha=0.5) # plot track
				ax1.set_aspect(1);ax2.set_aspect(1)
				ax1.set_title('Frames');ax2.set_title('Track')
				ax1.set_ylabel('Lat');ax1.set_xlabel('Lon')
				ax2.set_ylabel('Lat');ax2.set_xlabel('Lon')
				F.savefig('%sAOI-%s_%s' % \
					(figpath,AOIname,prevDate))
			except:
				pass
			# Create new figure for each new date
			F=plt.figure()
			ax1=F.add_subplot(1,2,1)
			ax1.fill(frameX,frameY,color=(0.6,0.6,0.6),alpha=0.5)
			ax2=F.add_subplot(1,2,2)
			# Create new polygon for each new date
			Track=FrameCorners
		prevDate=AcqDate # update for next iteration of loop
# --- Save final AOI ---
# Save track to kml
TrackName='AOI-%s_%s' % (AOIname,prevDate)
T=Polygon2KML(name=TrackName,polygon=Track)
T.write(outFilename='%sAOI-%s_%s.kml' % \
	(kmlpath,AOIname,prevDate))
# Save previous figure if exists
trackX,trackY=Track.exterior.xy # final track corners
ax2.fill(trackX,trackY,color='b',alpha=0.5) # plot track
ax1.set_aspect(1);ax2.set_aspect(1)
ax1.set_title('Frames');ax2.set_title('Track')
ax1.set_ylabel('Lat');ax1.set_xlabel('Lon')
ax2.set_ylabel('Lat');ax2.set_xlabel('Lon')
F.savefig('%sAOI-%s_%s' % \
	(figpath,AOIname,prevDate))

# Final stats
nFrames=len(granuleList) # valid frames within lat bounds
uniqueDates=[]
[uniqueDates.append(d) for d in frameDates if d not in uniqueDates]
nDatesUnique=len(uniqueDates)
print('Parsed %i frames total' % (nFramesTotal))
print('%i frames within Lat bounds: %f, %f' % \
	(nFrames,minLat,maxLat))
print('%i unique dates' % (nDatesUnique))