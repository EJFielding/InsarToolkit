#!/usr/bin/env python3
import glob
from InsarFormatting import ARIAname, listUnique

'''
Find and list the unique dates for a list of ARIA standard products
'''

### --- Parser ---
def createParser():
	import argparse
	parser = argparse.ArgumentParser(description='List unique dates for list of ARIA standard products.')
	parser.add_argument('-f','--folder',dest='fileFldr',type=str,default=None,help='Specify folder with ARIA products.')
	parser.add_argument('-l','--list',dest='fileList',type=str,default=None,help='Specify text file with ARIA product names.')
	parser.add_argument('-v','--verbose',dest='verbose',action='store_true',default=False,help='Verbose mode.')
	parser.add_argument('-o','--output',dest='outputList',type=str,default=None,help='Filename to write all unique dates.')
	return parser 

def cmdParser(inpt_args=None):
	parser = createParser()
	return parser.parse_args(inpt_args)



### --- Main function ---
if __name__=="__main__":
	inpt=cmdParser()

	## Load in data
	# For products in folder
	if inpt.fileFldr:
		Names=glob.glob('{}/*.nc'.format(inpt.fileFldr))
		Nproducts=len(Names) # number of products

	# For product names in list
	if inpt.fileList:
		Flist=open(inpt.fileList,'r') # open file list
		Flines=Flist.readlines() # read in lines
		Flist.close() # close

		# List of dates only or ARIA -count output?
		if Flines[0][:5]=='https':
			print('Reading ARIA output list...')
			Flines=Flines[1:-2] # remove formatting lines
			for i in range(len(Flines)):
				# Remove excess formatting
				Flines[i]=Flines[i].strip('Found: ') 

		# Format into product names
		Nproducts=len(Flines) # number of products
		Names=[] # empty list for product names
		[Names.append(n.strip('\n')) for n in Flines] # strip newline marker

	if inpt.verbose is True:
		print('Product names {}'.format(Names))
	print('{} products found'.format(Nproducts))

	## Create list of all dates
	AllDates=[]
	for i in range(Nproducts):
		# Parse name with ARIAname object
		ariaName=ARIAname(Names[i]) # instantiate
		AllDates.append(ariaName.RefDate) # add reference date to list
		AllDates.append(ariaName.SecDate) # add secondary date to list

	# Report
	Ndates=len(AllDates) # number of dates from product names
	if inpt.verbose is True:
		print('All dates {}'.format(AllDates))
	print('{} total dates detected'.format(Ndates))

	## List unique dates
	# Use listUnique function from InsarToolkit
	UniqueDates=listUnique(AllDates) # unique dates

	# Report
	Nunique=len(UniqueDates)
	if inpt.verbose is True:
		print('Unique dates {}'.format(UniqueDates))
	print('{} unique dates detected'.format(Nunique))

	## Save if requested
	if inpt.outputList:
		Fout=open(inpt.outputList,'w')
		for i in range(Nunique):
			Fout.write('{}\n'.format(UniqueDates[i]))
		Fout.close()