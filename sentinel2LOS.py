#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 


### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='Convert ground displacements to satellite line of sight.')
	# Arguments
	parser.add_argument('--heading',dest='heading',type=float,default=None,help='Flight direction of the spacecraft w.r.t. north (degrees)')
	parser.add_argument('--alpha',dest='alpha',type=float,default=None,help='Azimuth angle, ccw from east and the look direction from the ground target to the sensor (degrees)')
	parser.add_argument('--theta',dest='theta',type=float,default=None,help='Incidence angle w.r.t. nadir (degrees)')
	parser.add_argument('--P',dest='P',type=str,default=None,help='Pointing vector string \'x y z\'')
	parser.add_argument('-E','--east',dest='E',type=float,default=None,help='Ground displacement in eastward direction')
	parser.add_argument('-N','--north',dest='N',type=float,default=None,help='Ground displacement in northward direction')
	parser.add_argument('-Z','--z',dest='Z',type=float,default=None,help='Ground displacement in vertical direction')
	# Highly optional
	parser.add_argument('-v','--verbose',dest='verbose',action='store_true',default=False,help='Verbose mode')
	return parser 

def cmdParser(inpt_args=None):
	parser = createParser()
	return parser.parse_args(inpt_args)


### --- Functions ---
# Convert heading to azimuth angle
def heading2azimuth(heading,verbose=False):
	'''
	INPUTS
		heading is the flight direction of the spacecraft w.r.t north in degrees
	OUTPUTS
		azimuth angle is the counterclockwise angle between the East and 
		  the look direction from the ground target to the sensor (degrees)
	'''
	alpha=-(heading%180)

	if verbose is True:
		print('Heading: {:3.1f}\nAzimuth angle: {:3.1f}'.format(heading,alpha))

	return alpha


# Satellite orientation to line of sight
def orient2LOS(theta,alpha,E,N,Z):
	'''
	INPUTS
		theta is the incidence angle w.r.t. nadir
		alpha is the azimuth angle, defined as counterclockwise angle between the East and 
		  the look direction from the ground target to the sensor
		E is the eastward component of motion
		N is the northward component of motion
		Z is the upward component of motion
	'''
	# Convert to radians
	theta=np.deg2rad(theta) # look angle
	alpha=np.deg2rad(alpha) # azimuth angle

	# Project along angles
	LOS=Z*np.cos(theta)\
	+E*np.sin(theta)*np.cos(alpha)\
	+N*np.sin(theta)*np.sin(alpha)

	return LOS


# Convert satellite orientation to pointing vector
def orient2pointing(theta,alpha,verbose=False):
	'''
	INPUTS
		theta is the incidence angle w.r.t. nadir
		alpha is the azimuth angle, defined as counterclockwise angle between the East and 
		  the look direction from the ground target to the sensor
	OUTPUTS
		P is the 1x3 pointing vector [px, py, pz]
	'''
	# Convert to radians
	theta=np.deg2rad(theta) # look angle
	alpha=np.deg2rad(alpha) # azimuth angle

	# Convert angles to vector
	pz=np.cos(theta)
	horz=np.sin(theta)

	px=np.cos(alpha)*horz # <-- check these
	py=np.sin(alpha)*horz # <-- check these

	# Form vector
	P=np.array([px,py,pz])

	# Scale to unit length
	mag=np.sqrt(px**2+py**2+pz**2)
	P/=mag # scale

	if verbose is True:
		print('Pointing vector: x {}, y {}, z {}'.format(P[0],P[1],P[2]))

	print('Does not work yet')
	exit()

	return P 


# Pointing vector to line of sight
def pointing2LOS(P,U):
	'''
	INPUTS
		P is the pointing vector
		  composed of 3 components: P_E, P_N, P_Z
		  components will be normalized to unit values if necessary
		U is the displacement vector
		  composed of 3 components: U_E, U_N, U_Z
	'''

	# Scale P to unit length
	mag=np.linalg.norm(P,2) # magnitude = L2 norm
	P/=mag # scale magnitude

	# Check dimensions
	P=P.reshape(1,3)
	U=U.reshape(3,1)

	# Project U along P
	LOS=np.dot(P,U).squeeze()
	return LOS


### --- Main ---
if __name__ == "__main__":
	inpt=cmdParser()

	## Check inputs
	# If using pointing vector or orientation angles
	if inpt.P is not None and inpt.theta is not None:
		print('Specify either pointing vector or orientation parameters, not both!')
		exit()

	# Check only one of heading and alpha is specified
	if inpt.heading is not None and inpt.alpha is not None:
		print('Specify either heading or alpha, not both!')
		exit()

	# If heading is specified, convert to azimuth
	if inpt.heading is not None:
		inpt.alpha=heading2azimuth(inpt.heading,verbose=inpt.verbose)

	## Project into line of sight (LOS)
	# using satellite orientation angles
	if inpt.theta is not None:
		LOS=orient2LOS(theta=inpt.theta,alpha=inpt.alpha,
			E=inpt.E, N=inpt.N, Z=inpt.Z)

	# using pointing vector
	if inpt.P is not None:
		# Format P into vector
		P=np.array([float(p) for p in inpt.P.split()])
		# Format U into vector
		U=np.array([inpt.E,inpt.N,inpt.Z])

		# Project
		LOS=pointing2LOS(P=P,U=U)

	# Print if specified
	if inpt.verbose is True:
		# Determine net ground displacement
		uNet=np.sqrt(inpt.E**2+inpt.N**2+inpt.Z**2)

		print('Orig displacement vector: [{} {} {}]'.format(inpt.E,inpt.N,inpt.Z))
		print('Displacement magnitude: {}'.format(uNet))
		print('LOS motion: {}'.format(LOS))