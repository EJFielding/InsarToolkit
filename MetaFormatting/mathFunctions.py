# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic, general mathematical functions
# 
# By Rob Zinke 2019 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules 
import numpy as np 

# --- Degrees to meters --- 
def deg2m(degrees): 
	# Convert 
	m=degrees*111.111  
	return m 


#############################
### --- Trend fitting --- ###
#############################

# --- Linear trend ---
# Combination of sinusoids, linear trend, and offset
#  A*t + B
class fit_linear_trend:
	def __init__(self,t,y):
		# Basic parameters
		n=len(t) # number of data points
		self.t=t 

		# Design matrix
		#  Gb = y
		G=np.ones((n,2)) # all ones
		G[:,0]=t 
		self.G=G # save design matrix

		# Invert for parameters
		#  b = Ginv y ~= (GTG)-1 GT y
		self.beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(y)
		self.linear=self.beta[0] # linear slope
		self.offset=self.beta[1] # DC offset

	def reconstruct(self,t=None):
		if t is not None:
			# Use new timepoints
			n=len(t) # new length
			G=np.ones((n,2)) # reconstruct design matrix
			G[:,0]=t 
		else:
			# Use old timepoints
			G=self.G

		# Estimate timeseries
		self.yhat=np.dot(G,self.beta)


# --- Periodic trend ---
# Combination of sinusoids, linear trend, and offset
#  A*cos(t) + B*sin(t) + C*t + D
class fit_periodic_trend:
	def __init__(self,t,y):
		# Basic parameters
		n=len(t) # number of data points
		self.t=t 

		# Design matrix
		#  Gb = y
		G=np.ones((n,4)) # all ones
		G[:,0]=np.cos(2*np.pi*t) 
		G[:,1]=np.sin(2*np.pi*t)
		G[:,2]=t 
		self.G=G # save design matrix

		# Invert for parameters
		#  b = Ginv y ~= (GTG)-1 GT y
		self.beta=np.linalg.inv(np.dot(G.T,G)).dot(G.T).dot(y)
		self.cosine=self.beta[0] # cosine coefficient
		self.sine=self.beta[1] # sine coefficient
		self.linear=self.beta[2] # linear slope
		self.offset=self.beta[3] # DC offset
		self.periodic=np.sqrt(self.cosine**2+self.sine**2) # magnitude of periodic signal

	def reconstruct(self,t=None):
		if t is not None:
			# Use new timepoints
			n=len(t) # new length
			G=np.ones((n,4)) # reconstruct design matrix
			G[:,0]=np.cos(2*np.pi*t) 
			G[:,1]=np.sin(2*np.pi*t)
			G[:,2]=t 
		else:
			# Use old timepoints
			G=self.G

		# Estimate timeseries
		self.yhat=np.dot(G,self.beta)