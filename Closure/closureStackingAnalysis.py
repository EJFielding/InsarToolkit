#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Analyze results of interferogram stacking computed 
#  using the "closureFromStacking.py" routine
# 
# Rob Zinke 2019
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
from dateFormatting import daysBetween, cumulativeTime
from viewingFunctions import imagettes, mapStats, imgBackground
from mathFunctions import fit_linear_trend, fit_periodic_trend


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Compare differences in phase velocity from HDF5 files.')
	# Folder with dates
	parser.add_argument('-n1', dest='n1File', type=str, help='n-n1 dataset')
	parser.add_argument('-n2', dest='n2File', type=str, help='n-n2 dataset')
	parser.add_argument('-y1', dest='y1File', type=str, help='n-y1 dataset')
	parser.add_argument('-y2', dest='y2File', type=str, help='n-y2 dataset')
	# Date/time criteria
	# Ancillary plotting
	parser.add_argument('--plotInputs', dest='plotInputs', action='store_true', help='Plot inputs')
	# Trend fitting
	parser.add_argument('--fit-linear', dest='fitLinear', action='store_true', help='Fit linear trend')
	parser.add_argument('--fit-periodic', dest='fitPeriodic', action='store_true', help='Fit periodic + linear trend')
	# Outputs
	parser.add_argument('-v','--verbose',dest='verbose', action='store_true', help='Verbose mode')
	#parser.add_argument('-o','--outName',dest='outName', type=str, default=None, help='Output name')
	return parser

def cmdParser(iargs = None):
	parser = createParser()
	return parser.parse_args(args=iargs)


### --- Formatting and processing ---
def cumulativePhaseCompare(event):
	# Location
	px=event.xdata; py=event.ydata
	px=int(round(px)); py=int(round(py))

	# Mean misclosure
	meanValue=meanDifference[py,px]
	print('px {} py {} = (mean) {}'.format(px,py,meanValue))

	# Clear current axes
	axProfs.cla()

	# Compare cumulative phase values
	N1phase=N1['cumulativePhase'][:,py,px]
	N2phase=N2['cumulativePhase'][:,py,px]
	Y1phase=Y1['cumulativePhase'][:,py,px]
	Y2phase=Y2['cumulativePhase'][:,py,px]

	# Plot data
	axProfs.plot(N1times,N1phase,marker=None,color='k',zorder=2,label='n-n1')
	axProfs.plot(N2times,N2phase,marker=None,color='b',zorder=4,label='n-n2')
	#axProfs.plot(Y1times,Y1phase,marker='.',color='m',zorder=6,label='n-yr1')
	#axProfs.plot(Y2times,Y2phase,marker='.',color='g',zorder=8,label='n-yr2')

	# Fit trend
	t=np.linspace(np.min(N1times),np.max(N1times),1000)
	if inpt.fitLinear is True:
		# Fit pure linear trend
		N1fit=fit_linear_trend(N1times,N1phase); N1fit.reconstruct(t)
		N2fit=fit_linear_trend(N2times,N2phase); N2fit.reconstruct(t)
		#Y1fit=fit_linear_trend(Y1times,Y1phase); Y1fit.reconstruct(t)
		#Y2fit=fit_linear_trend(Y2times,Y2phase); Y2fit.reconstruct(t)

		# Plot pure linear trend
		axProfs.plot(t,N1fit.yhat,'k',linewidth=2,zorder=1,alpha=0.5)
		axProfs.plot(t,N2fit.yhat,'b',linewidth=2,zorder=3,alpha=0.5)

	elif inpt.fitPeriodic is True:
		# Fit periodic + linear trend
		N1fit=fit_periodic_trend(N1times,N1phase); N1fit.reconstruct(t)
		N2fit=fit_periodic_trend(N2times,N2phase); N2fit.reconstruct(t)
		#Y1fit=fit_periodic_trend(Y1times,Y1phase); Y1fit.reconstruct(t)
		#Y2fit=fit_periodic_trend(Y2times,Y2phase); Y2fit.reconstruct(t)

		# Plot periodic + linear trend
		axProfs.plot(t,N1fit.yhat,'k',linewidth=2,zorder=1,alpha=0.5)
		axProfs.plot(t,N2fit.yhat,'b',linewidth=2,zorder=3,alpha=0.5)


	axProfs.set_xlabel('time since {} (years)'.format(refDate))
	axProfs.set_ylabel('cumulative phase (rad)')
	axProfs.legend()

	Fprofs.canvas.draw()


### --- Main function ---
if __name__=='__main__':
	# Gather arguments
	inpt=cmdParser()

	# Load datasets
	N1=h5py.File(inpt.n1File)
	N2=h5py.File(inpt.n2File)
	Y1=h5py.File(inpt.y1File)
	Y2=h5py.File(inpt.y2File)

	# Dates
	N1dates=[pair[0] for pair in N1['datePairs']]
	N2dates=[pair[0] for pair in N2['datePairs']]
	Y1dates=[pair[0] for pair in Y1['datePairs']]
	Y2dates=[pair[0] for pair in Y2['datePairs']]
	refDate=np.min(N1dates[0])

	# Time elapsed
	N1times=np.array([daysBetween(refDate,date)/365.25 for date in N1dates])
	N2times=np.array([daysBetween(refDate,date)/365.25 for date in N2dates])
	Y1times=np.array([daysBetween(refDate,date)/365.25 for date in Y1dates])
	Y2times=np.array([daysBetween(refDate,date)/365.25 for date in Y2dates])

	# Plot inputs if requested
	if inpt.plotInputs is True:
		imagettes(N1['cumulativePhase'],3,4,cmap='jet',pctmin=2,pctmax=98,background='auto',colorbarOrientation='horizontal',
			supTitle='n-n1 cumulative phase',titleList=N1['cumulativeTime'])
		imagettes(N2['cumulativePhase'],3,4,cmap='jet',pctmin=2,pctmax=98,background='auto',colorbarOrientation='horizontal',
			supTitle='n-n2 cumulative phase',titleList=N2['cumulativeTime'])
		imagettes(Y1['cumulativePhase'],3,4,cmap='jet',pctmin=2,pctmax=98,background='auto',colorbarOrientation='horizontal',
			supTitle='n-y1 cumulative phase',titleList=Y1['cumulativeTime'])
		imagettes(Y2['cumulativePhase'],3,4,cmap='jet',pctmin=2,pctmax=98,background='auto',colorbarOrientation='horizontal',
			supTitle='n-y2 cumulative phase',titleList=Y2['cumulativeTime'])

	## Calculate average LOS phase velocities
	# Mean velocity = total phase change / total time
	N1avePhaseVelocity=N1['cumulativePhase'][-1]/N1['cumulativeTime'][-1]
	N2avePhaseVelocity=N2['cumulativePhase'][-1]/N2['cumulativeTime'][-1]
	Y1avePhaseVelocity=Y1['cumulativePhase'][-1]/Y1['cumulativeTime'][-1]
	Y2avePhaseVelocity=Y2['cumulativePhase'][-1]/Y2['cumulativeTime'][-1]

	# Plot average phase velocities
	F=plt.figure()

	ax=F.add_subplot(2,2,1) # n-n1
	N1map=np.ma.array(N1avePhaseVelocity,mask=(N1avePhaseVelocity==N1avePhaseVelocity[0,0]))
	N1stats=mapStats(N1map,pctmin=2,pctmax=98)
	ax.set_title('n-n1')
	cax=ax.imshow(N1map,cmap='jet',vmin=N1stats.vmin,vmax=N1stats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='horizontal')

	ax=F.add_subplot(2,2,2) # n-n2
	N2map=np.ma.array(N2avePhaseVelocity,mask=(N2avePhaseVelocity==N2avePhaseVelocity[0,0]))
	N2stats=mapStats(N2map,pctmin=2,pctmax=98)
	ax.set_title('n-n2')
	cax=ax.imshow(N2map,cmap='jet',vmin=N2stats.vmin,vmax=N2stats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='horizontal')

	ax=F.add_subplot(2,2,3) # n-y1
	Y1map=np.ma.array(Y1avePhaseVelocity,mask=(Y1avePhaseVelocity==Y1avePhaseVelocity[0,0]))
	Y1stats=mapStats(Y1map,pctmin=2,pctmax=98)
	ax.set_title('n-y1')
	cax=ax.imshow(Y1map,cmap='jet',vmin=Y1stats.vmin,vmax=Y1stats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='horizontal')

	ax=F.add_subplot(2,2,4) # n-y2
	Y2map=np.ma.array(Y2avePhaseVelocity,mask=(Y2avePhaseVelocity==Y2avePhaseVelocity[0,0]))
	Y2stats=mapStats(Y2map,pctmin=2,pctmax=98)
	ax.set_title('n-y2')
	cax=ax.imshow(Y2map,cmap='jet',vmin=Y2stats.vmin,vmax=Y2stats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='horizontal')

	F.suptitle('Mean LOS phase velocity by pair')


	## Velocity differences
	# Compute velocity differences
	N2diff=N1avePhaseVelocity-N2avePhaseVelocity
	Y1diff=N1avePhaseVelocity-Y1avePhaseVelocity
	Y2diff=N1avePhaseVelocity-Y2avePhaseVelocity

	# Plot velocity differences
	F=plt.figure()

	ax=F.add_subplot(3,1,1) # n-n2
	N2diffmap=np.ma.array(N2diff,mask=(N2diff==N2diff[0,0]))
	N2diffstats=mapStats(N2diffmap,pctmin=2,pctmax=98)
	ax.set_title('nn1-nn2')
	cax=ax.imshow(N2diffmap,cmap='jet',vmin=N2diffstats.vmin,vmax=N2diffstats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='vertical')

	ax=F.add_subplot(3,1,2) # n-y1
	Y1diffmap=np.ma.array(Y1diff,mask=(Y1diff==Y1diff[0,0]))
	Y1diffstats=mapStats(Y1diffmap,pctmin=2,pctmax=98)
	ax.set_title('nn1-ny1')
	cax=ax.imshow(Y1diffmap,cmap='jet',vmin=Y1diffstats.vmin,vmax=Y1diffstats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='vertical')

	ax=F.add_subplot(3,1,3) # n-y2
	Y2diffmap=np.ma.array(Y2diff,mask=(Y2diff==Y2diff[0,0]))
	Y2diffstats=mapStats(Y2diffmap,pctmin=2,pctmax=98)
	ax.set_title('nn1-ny2')
	cax=ax.imshow(Y2diffmap,cmap='jet',vmin=Y2diffstats.vmin,vmax=Y2diffstats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	F.colorbar(cax,orientation='vertical')

	F.suptitle('Difference by pair')


	## Mean of all velocity maps
	# meanDifference=(N2diff+Y1diff+Y2diff)/3
	meanDifference=N2diff

	# Plot mean difference
	meanDifference=np.ma.array(meanDifference,mask=(meanDifference==meanDifference[0,0]))
	stats=mapStats(meanDifference,pctmin=2,pctmax=98)

	Fmean=plt.figure()
	ax=Fmean.add_subplot(111)
	cax=ax.imshow(meanDifference,cmap='jet',vmin=stats.vmin,vmax=stats.vmax)
	ax.set_xticks([]); ax.set_yticks([])
	ax.set_title('(n-n1) - (n-n2) difference')
	Fmean.colorbar(cax,orientation='horizontal')

	## Click for profiles
	Fprofs=plt.figure()
	axProfs=Fprofs.add_subplot(111)

	Fmean.canvas.mpl_connect('button_press_event',cumulativePhaseCompare)


	plt.show()