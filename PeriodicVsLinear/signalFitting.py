#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from dateFormatting import daysBetween
from viewingFunctions import imgBackground
from mathFunctions import fit_linear_trend, fit_periodic_trend


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Fit trends to pixels in data cube.')
	# Data cube file
	parser.add_argument(dest='fName', type=str, help='HDF5 data cube')

	# Displacement map options
	parser.add_argument('-i','--index2show', dest='index2show', type=int, default=-1, help='Index of epoch to plot')
	parser.add_argument('-d','--date2show', dest='date2show', type=int, default=None, help='Date to show (YYYYMMDD)')
	parser.add_argument('--additional-nodata', dest='addtNodata', type=float, default=None, help='Additional no data value')
	parser.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis', help='Colormap')
	parser.add_argument('--cbar-orientation', dest='cbarOrientation', type=str, default='vertical', help='Colorbar orientation')
	parser.add_argument('--pctmin', dest='pctmin', type=float, default=0, help='Minimum percentile to clip')
	parser.add_argument('--pctmax', dest='pctmax', type=float, default=100, help='Maximum percentile to clip')
	parser.add_argument('--vmin', dest='vmin', type=float, default=None, help='Minimum displacement value to plot')
	parser.add_argument('--vmax', dest='vmax', type=float, default=None, help='Maximum displacement value to plot')

	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Plotting functions ---
# Plot location on map
def plotLocation(event):
	# Get location
	px=event.xdata; py=event.ydata
	px=int(round(px)); py=int(round(py))

	if inpt.verbose is True:
		print('Location: x {}; y {}'.format(px,py))
	
	axMap.cla()
	caxMap=axMap.imshow(Ushow,
		cmap='jet',vmin=inpt.vmin,vmax=inpt.vmax,
		zorder=1)
	axMap.plot(px,py,'ko',zorder=2)

	Fmap.canvas.draw()


	# Get data
	u=U[:,py,px]

	# Plot raw data
	axData.cla()
	axData.plot(times,u,'-ko')
	axData.set_xlabel('time (years)')
	axData.set_ylabel('displacement (m)')

	# Plot periodic trend
	periodicTrend=fit_periodic_trend(times,u)
	periodicTrend.reconstruct()
	P=periodicTrend.periodic # periodic coefficient
	L=np.abs(periodicTrend.linear) # linear coefficient

	pctPeriodic=P/(P+L)
	pctLinear=L/(P+L)

	R=P/L

	axData.plot(times,periodicTrend.yhat,'b')
	axData.set_title('Ratio: {:.2f}; Periodic: {:.2f}; Linear: {:.2f}'.\
		format(R,pctPeriodic,pctLinear))

	Fdata.canvas.draw()



### --- Main function ---
if __name__=='__main__':
	## Gather arguments
	inpt=cmdParser()


	## Load data cube
	DS=h5py.File(inpt.fName,'r')


	# Parse data sets
	dates=list(DS['date'][:].astype(int))
	U=DS['timeseries'][:,:,:].astype(float)

	if inpt.date2show:
		inpt.index2show=dates.index(inpt.date2show)
	else:
		inpt.date2show=dates[inpt.index2show]


	if inpt.verbose is True:
		print('Loaded: {}'.format(inpt.fName))
		print(DS.keys())
		print('Dates: {}'.format(dates))
		print('Timeseries array size: {}'.format(U.shape))
		print('Displaying date: ({}) {}'.format(inpt.index2show,inpt.date2show))


	## Calculate time since beginning
	refTime=dates[0] 
	times=np.array([daysBetween(refTime,date,absTime=False)/365.25 for date in dates])


	## Plot topmost layer of data cube
	# Map figure
	Ushow=U[inpt.index2show,:,:]
	bg=imgBackground(Ushow)
	Ushow=np.ma.array(Ushow,mask=(Ushow==bg) | (Ushow==inpt.addtNodata))

	if not inpt.vmin:
		inpt.vmin=np.percentile(Ushow.compressed(),inpt.pctmin)
	if not inpt.vmax:
		inpt.vmax=np.percentile(Ushow.compressed(),inpt.pctmax)

	Fmap=plt.figure()
	axMap=Fmap.add_subplot(111)
	caxMap=axMap.imshow(Ushow,
		cmap='jet',vmin=inpt.vmin,vmax=inpt.vmax,
		zorder=1)
	Fmap.colorbar(caxMap,orientation=inpt.cbarOrientation)


	## Gather  and plot data
	# Data figure
	Fdata=plt.figure()
	axData=Fdata.add_subplot(111)
	Fmap.canvas.mpl_connect('button_press_event',plotLocation)


	plt.show()