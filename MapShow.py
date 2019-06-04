#!/usr/bin/env python3 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find and plot the difference in phase between two 
#  wrapped interferograms 
# 
# by Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys 
import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal 


# --- Help menu --- 
if sys.argv.count('--help')>0: 
	print(''' 
INPUTS 
	--dsImg 	[intiger]		Power-of-2 downsample factor
								(does not affect statistics) 

	--vmin 		[float] 		Min display value 
	--vmax 		[float]			Max display value 
	--pctmin 	[float] 		Min value percent (not decimal)
	--pctmax 	[float] 		Max value percent (not decimal) 

	--BG 		[opt:float]		Background value 

	--hist 		[opt:integer]	Histogram 
		''') 
	exit() 


# --- Load image --- 
img_name=sys.argv[1] 

img=gdal.Open(img_name,gdal.GA_ReadOnly) 
nB=img.RasterCount 

if nB==1:
	B=1 # Wrapped 
elif nB==2: 
	B=2 # Unwrapped 
print('Bands: %i' % B) 

I=img.GetRasterBand(B).ReadAsArray() 


# --- Formatting --- 
# Image type (real/complex)
print('Type: %s' % type(I[0,0])) 
if isinstance(I[0,0],np.complex64):
	Imag=np.abs(I) 
	I=np.angle(I) 

# Statistics 
Iarray=I.reshape(-1,1) 

# Background 
if sys.argv.count('--BG')>0: 
	n_BG=sys.argv.index('--BG') 
	try: 
		BG=float(sys.argv[n_BG+1]) 
	except: 
		BG=0 
	I=np.ma.array(I,mask=(I==BG)) 
	Iarray=Iarray[Iarray!=BG] 
	print('Removed BG: %f' % (BG))


vmin=None 
vmax=None 
if sys.argv.count('--vmin')>0: 
	n_vmin=sys.argv.index('--vmin') 
	vmin=sys.argv[n_vmin+1]
if sys.argv.count('--vmax')>0: 
	n_vmax=sys.argv.index('--vmax') 
	vmax=sys.argv[n_vmax+1] 
if sys.argv.count('--pctmin')>0:
	n_pctmin=sys.argv.index('--pctmin') 
	vmin=np.percentile(Iarray,float(sys.argv[n_pctmin+1])) 
if sys.argv.count('--pctmax')>0: 
	n_pctmax=sys.argv.index('--pctmax') 
	vmax=np.percentile(Iarray,float(sys.argv[n_pctmax+1])) 
print('vmin:',vmin) 
print('vmax:',vmax) 

if vmin is not None and vmax is not None:
	Iarray=Iarray[(Iarray>=float(vmin)) & (Iarray<=float(vmax))] 

# Histogram 
if sys.argv.count('--hist')>0:
	n_hist=sys.argv.index('--hist') 
	try:
		nbins=int(sys.argv[n_hist+1]) 
	except:
		nbins=100 
	H,edges=np.histogram(Iarray,nbins) 
	centers=edges[:-1]+np.diff(edges)/2 

	Hist=plt.figure('Hist') 
	axH=Hist.add_subplot(111) 
	axH.plot(centers,H,'b') 
	axH.set_title('Histogram (%i bins)' % (nbins))  

# Downsample image 
if sys.argv.count('--dsImg')>0:
	n_dsImg=sys.argv.index('--dsImg')
	dsImg=int(sys.argv[n_dsImg+1]) 
	dsImg=2**dsImg 
else:
	dsImg=1 


# --- Main plot --- 
F=plt.figure() 
ax=F.add_subplot(111) 
cax=ax.imshow(I[::dsImg,::dsImg],vmin=vmin,vmax=vmax)
F.colorbar(cax,orientation='horizontal') 


plt.show() 
