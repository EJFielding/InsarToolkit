# --- Unique values ---
# Find the unique values in a list
def listUnique(L):
	uniqueVals=[] # empty list for unique values
	# Loop through each value and add to list if not there
	[uniqueVals.append(i) for i in L if i not in uniqueVals] 
	return uniqueVals