"""
	Functions related to calculating mass movement from a DEM
	 and LOS InSAR observations.
"""

### IMPORT MODULES ---
import numpy as np
from scipy.signal import convolve2d


### GENERIC ANGLES AND VECTORS ---
## Convert vector to angle
def vect2az(v):
	"""
		Convert a vector to an angle in compass azimuth (0/360 = north)
		INPUTS
			v is a vector in np.array form, with 
			 shape (2, ), (1,2), or (2,1)
		OUTPUTS
			az is the compass azimuth in degress 
	"""

	# Flatten to 1d array
	v=v.flatten()

	# Normalize vector to unit length
	v=v/np.linalg.norm(v,2)

	# Compute angle
	ang=np.arctan2(v[1],v[0])

	# Convert radians to degrees
	ang=180*ang/np.pi

	# Compass direction
	az=90-ang
	if az<0: az=360+az

	return az


## Convert azimuth to vector
def az2vect(az):
	"""
		Convert compass azimuth to a vector
		INPUTS
			az is the compass azimuth in degrees (0/360 = north)
		OUTPUTS
			v is the vector representation expressed as a (2, )
			 numpy array, where v = [east, north]
	"""

	# Convert degrees to radians
	ang=np.pi/180*ang

	# Convert from azimuth to Cartesian
	ang=np.pi/2-ang
	if ang<0: ang=2*np.pi+ang

	# Compute vector
	x=np.cos(ang); y=np.sin(ang) # components
	v=np.array([x,y]) # construct vector

	# Vector to unit length
	v=v/np.linalg.norm(v,2)

	return v


## 3D vector to azimuth, slope angles
def vect3d2orient(vx,vy,vz):
	"""
		Convert 3D vector components to azimuth and slope
		INPUTS
			vx, vy, vz are the 3D vector components
		OUTPUTS
			azimuth, slope in degrees
	"""

	# Compute azimuth angle
	azimuth=np.arctan2(vy,vx)
	azimuth=np.rad2deg(azimuth)
	azimuth=90-azimuth

	# Compute slope angle
	slope=np.arctan(vz/np.sqrt(vx**2+vy**2))
	slope=np.rad2deg(slope)

	return azimuth, slope


## Azimuth, slope angles to 3D vector
def orient2vect3d(azimuth,slope):
	"""
		Convert azimuth and slope angles in degrees to 3D
		 vector components
		INPUTS
			azimuth in degrees clockwise from north
			slope in degrees from horizontal
		OUTPUTS
			v is the 3D unit vector representation comprising
			 [vx, vy, vz]
	"""

	# Convert angles to radians
	azimuth=90-azimuth
	azimuth=np.deg2rad(azimuth)
	slope=np.deg2rad(slope)

	# Compute vertical and horizontal components
	vz=np.sin(slope)
	horz=np.cos(slope)

	# Compute azimuth components
	vx=np.cos(azimuth)*horz
	vy=np.sin(azimuth)*horz

	# Normalize to unit length
	vmag=np.sqrt(vx**2+vy**2+vz**2)
	vx/=vmag
	vy/=vmag
	vz/=vmag

	return vx,vy,vz


## Satellite geometry to 3D vector
def satGeom2vect3d(satAz,satLook):
	"""
		Convert satellite geometry (azimuthAngle and look/incidenceAngle)
		 into 3D vector components. Following the ARIA convention, azimuth
		 angle simulates the target looking at the satellite, measured in 
		 degrees from east. Look angle is the angle between vertical and
		 the satellite look direction. Incidence angle is between the 
		 normal to the spheroid and the satellite look direction.
		 This formulation works for both lookAngle and incidenceAngle. 
	"""

	# Convert angles to radians
	satAz=np.deg2rad(satAz)
	satLook=np.deg2rad(satLook)

	# Horizontal components
	vx=np.cos(satAz)
	vy=np.sin(satAz)

	# Vertical component
	vz=np.cos(satLook)

	# Scale horizontal components
	h=np.sin(satLook)
	vx*=h
	vy*=h

	# Flip sign to point from satellite to target
	vx=-vx
	vy=-vy
	vz=-vz

	# Scale to unit length
	L=np.sqrt(vx**2+vy**2+vz**2)
	vx/=L
	vy/=L
	vz/=L

	return vx,vy,vz



### CONVERT DEM TO VECTOR FIELD ---
## Convert DEM to slope, aspect gradients
def computeGradients(elev,dx,dy):
	"""
		Given a rasterized DEM, compute the slope and slope-aspect
		 maps in vector form. Be certain that the coordinates are 
		 projected into a Cartesian system (e.g., UTM). Explicitly 
		 specify the pixel sizes in east and north for mistake-proofing.
		INPUTS
			elev is a MxN array representing rasterized elevation
			 values
			dx is the pixel size in easting
			dy is the pixel size in northing
			(kernel) is the kernel type (Roberts, Prewitt, [Sobel], Scharr)
		OUTPUTS
			slope is the slope map in degrees from horizontal (0-90)
			azimuth is the azimuth map in degrees from north (0-360)
	"""

	# Define kernel type 
	sobel=np.array([[-1-1.j,0-2.j,1-1.j],
					[-2+0.j,0+0.j,2+0.j],
					[-1+1.j,0+2.j,1+1.j]]) 
	sobel.real=sobel.real/(8*dx); sobel.imag=sobel.imag/(8*dy) 

	# Calculate gradient map 
	gradients=convolve2d(elev,sobel,mode='same')

	return gradients


## Convert gradient to slope
def grad2slope(gradients):
	"""
		Gradient is a map of complex values representing the
		 gradient of topography real = dz/dx; imag = dz/dy
		Returns slope values in degrees
	"""

	# Compute slope
	dz=np.abs(gradients)
	slope=np.arctan(dz)
	slope=np.rad2deg(slope)

	return slope


## Convert gradient to slope-aspect
def grad2aspect(gradients):
	"""
		Gradient is a map of complex values representing the
		 gradient of topography real = dz/dx; imag = dz/dy
		Returns aspect values in degrees wrt north
	"""

	# Compute aspect
	aspect=np.arctan2(gradients.imag,gradients.real)
	aspect=np.rad2deg(aspect)
	aspect=90-aspect
	aspect[aspect<0]+=360; aspect=aspect%360

	return aspect


## Compute pointing vector maps
def makePointingVectors(gradients):
	"""
		Given a gradient map, compute the three-component unit
		 vector that points directly downhill. 

		pz is vertical component, from the partial derivatives
		h is the magnitude of the horizontal vectors
		 equivalent to z
		P = [x/h, y/h, z] = [x/z, y/z, z]
	"""

	# Compute pointing vector
	pz=np.abs(gradients); pz=pz.real
	px=gradients.real/pz; px=px.real
	py=gradients.imag/pz; py=py.real

	# Normalize length to 1.0
	L=np.sqrt(px**2+py**2+pz**2)
	px/=L
	py/=L
	pz/=L

	# Z-component is negative
	pz=-pz

	return px, py, pz


## Pointing vectors to slope
def pointing2slope(px,py,pz):
	"""
		(Re)Compute the slope map from vector components
		px, py, pz are vector fields (MxN maps), where x is the
		 east component of the gradient, y is the north component, 
		 and z is the vertical component, such that:
		P = [px, py, pz].T
		Slope ~ arctan(pz/||px,py||)
	"""
	
	# Compute magnitude of horizontal component
	H=np.sqrt(px**2+py**2)

	# Compute slope
	slope=np.arctan(pz/H)

	# Format slope values
	slope=np.rad2deg(slope)

	return slope


## Pointing vectors to aspect
def pointing2aspect(px,py,pz):
	"""
		(Re)Compute the aspect map from vector components
		px, py, pz are vector fields (MxN maps), where x is the
		 east component of the gradient, y is the north component, 
		 and z is the vertical component, such that:
		P = [px, py, pz].T
		Aspect ~ arctan(py/px)
		pz is not necessary for this computation, but is included
		 for consistency
	"""

	# Compute aspect
	aspect=np.arctan2(py,px)

	# Format aspect values
	aspect=np.rad2deg(aspect)
	aspect=90-aspect
	aspect[aspect<0]+=360; aspect=aspect%360

	return aspect



### NORMAL VECTORS ---
## Normal to a single vector
def normalVector(d1,d2,d3):
	"""
		This function computes the normal to the vector given, with
		 the same azimuth. The normal is calculated to satisfy the 
		 condition < n,d > = 0
		d is the 3-component vector that points up/down the hill slope
		n is the 3-component vector normal to d
	"""

	# Replicate azimuth of d-vector
	n=np.array([d1,d2,0])
	
	# Compute the vertical component of the normal vector
	n[2]=-(d1**1+d2**2)/d3

	# Normalize to unit length
	n=n/np.linalg.norm(n,2)

	return n


## Normals to pointing vectors
def normals2pointing(px,py,pz):
	"""
		This function computes the normal to the vector given, with
		 the same azimuth. The normal is calculated to satisfy the 
		 condition < n,d > = 0
		p is the 3-component vector that points along the 
		 topographic gradient
		n is the 3-component vector normal to d
	"""

	# Replicate azimuth of pointing vector
	nx=px.copy()
	ny=py.copy()

	# Compute vertical component of normal vector that satisfies
	#	n1p1 + n2p2 + n3p3 = 0
	nz=-(px**2+py**2)/pz

	# Normalize to unit length
	L=np.sqrt(nx**2+ny**2+nz**2)
	nx/=L
	ny/=L
	nz/=L

	return nx, ny, nz