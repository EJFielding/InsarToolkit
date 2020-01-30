import h5py

# --- Load HDF5 file ---


# --- Save HDF5 file ---
def saveHDF(outName,dataDict,verbose=False):
	out=h5py.File(outName,'w')
	if verbose is True:
		print('Saving to HDF5: {}'.format(outName))
	for key in dataDict.keys():
		out.create_dataset(key,data=dataDict[key])
		if verbose is True:
			print('... saving {}'.format(key))
	out.close()