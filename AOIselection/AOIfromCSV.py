#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import csv
from KMLwriter import Polygon2KML


### --- Parser ---
def createParser():
	'''
		Provide a list of triplets to investigate phase closure.
		Requires a list of triplets---Use "tripletList.py" script to
		 generate this list. 
	'''

	import argparse
	parser = argparse.ArgumentParser(description='Determine phase closure.')
	# Required inputs
	parser.add_argument('-f','--fname','--filename', dest='CSVname', type=str, required=True, help='CSV file')
	# Date/time criteria
	parser.add_argument('-maxLat','--max-Latitude', dest='maxLat', type=float, default=None, help='Maximum latitude')
	parser.add_argument('-minLat','--min-Latitude', dest='minLat', type=float, default=None, help='Minimum latitude')
	parser.add_argument('-maxFrames', '--max-frames', dest='maxFrames', type=float, default=None, help='Max number of frames')
	parser.add_argument('-selectDates', '--select-dates', dest='selectDates', default=None, help='List of dates to use in AOI')
	# Outputs
	parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose')
	parser.add_argument('-o','--outName', dest='outName', type=str, default=None, required=True, help='Saves maps to output')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


# --- Functions ---
class Frame:
	def __init__(self,granule):
		self.granule=granule


# --- Main ---
if __name__=='__main__':
	print('WARNING: Requires modification of "Beam Mode Description" in CSV')
	print('WARNING: Midnight crossing will likely be a problem')


	# Check files exist
	if not os.path.exists('KMLs'):
		os.mkdir('KMLs')
	if not os.path.exists('Figs'):
		os.mkdir('Figs')

	# Gather arguments
	inpt=cmdParser()

	## Load CSV file
	CSVfile=open(inpt.CSVname)
	CSVdict=csv.DictReader(CSVfile, delimiter=',')

	# Read rows in csv file
	frames={} # dictionary for storing frame instances
	nFrames=0 # start counting number of frames
	prevDate=0 # start keeping track of date
	for frame in CSVdict:
		nFrames+=1

		## Pull metadata for each frame
		# Check within lat bounds
		CenterLat=float(frame['Center Lat'])
		ValidLat=True
		if inpt.minLat is not None and (CenterLat<=inpt.minLat):
			ValidLat=False
		if inpt.maxLat is not None and (CenterLat>=inpt.maxLat):
			ValidLat=False
		# Check is selected date
		AcqDate=frame['Acquisition Date'] # e.g., 2015-12-08T00:11:30.000000
		AcqDate=AcqDate.split('T')[0] # date only, e.g., 2015-12-08
		AcqDate=AcqDate.replace('-','') # remove dashes, e.g., 20151208
		ValidDate=True
		if inpt.selectDates is not None:
			print('Use of a list of valid dates is not yet working!'); exit()
			if AcqDate not in selectDates:
				ValidDate=False
		# Instantiate if criteria met
		if ValidLat is True and ValidDate is True:
			granule=frame['Granule Name'] # granule name
			frames[granule]=Frame(granule) # instantiate frame
			# Gather spatial data
			frames[granule].CenterLat=CenterLat
			CenterLon=float(frame['Center Lon'])
			NearStartLat=float(frame['Near Start Lat'])
			NearStartLon=float(frame['Near Start Lon'])
			FarStartLat=float(frame['Far Start Lat'])
			FarStartLon=float(frame['Far Start Lon'])
			NearEndLat=float(frame['Near End Lat'])
			NearEndLon=float(frame['Near End Lon'])
			FarEndLat=float(frame['Far End Lat'])
			FarEndLon=float(frame['Far End Lon'])
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
					TrackName='AOI-{}_{}'.format(inpt.outName,prevDate)
					print('Track name: {}'.format(TrackName))
					T=Polygon2KML(name=TrackName,polygon=Track)
					T.write(outFilename='KMLs/AOI-{}_{}.kml'.format(inpt.outName,prevDate))
					# Save previous figure if exists
					trackX,trackY=Track.exterior.xy # final track corners
					ax2.fill(trackX,trackY,color='b',alpha=0.5) # plot track
					ax1.set_aspect(1);ax2.set_aspect(1)
					ax1.set_title('Frames');ax2.set_title('Track')
					ax1.set_ylabel('Lat');ax1.set_xlabel('Lon')
					ax2.set_ylabel('Lat');ax2.set_xlabel('Lon')
					F.savefig('Figs/AOI-{}_{}'.format(inpt.outName,prevDate))
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
	TrackName='AOI-{}_{}'.format(inpt.outName,prevDate)
	T=Polygon2KML(name=TrackName,polygon=Track)
	T.write(outFilename='KMLs/AOI-{}_{}.kml'.format(inpt.outName,prevDate))
	# Save previous figure if exists
	trackX,trackY=Track.exterior.xy # final track corners
	ax2.fill(trackX,trackY,color='b',alpha=0.5) # plot track
	ax1.set_aspect(1);ax2.set_aspect(1)
	ax1.set_title('Frames');ax2.set_title('Track')
	ax1.set_ylabel('Lat');ax1.set_xlabel('Lon')
	ax2.set_ylabel('Lat');ax2.set_xlabel('Lon')
	F.savefig('Figs/AOI-{}_{}'.format(inpt.outName,prevDate))

	# Final stats
	print('Parsed {} frames total'.format(nFrames))
	plt.show()