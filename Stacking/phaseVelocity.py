def phaseStackHDF(phaseCube,dates=None,storeCumulative=False):
	'''
		Phase is a 3D data cube for which the first dimension is 
		 layers, second dimension is NS dim, third dimension is EW
		Dates are optional, but if specified, the final output will
		 represent a mean rather than a total. 
		If "storeCumulative" is true, a 3D data cube will be
		 stored and returned.
	'''