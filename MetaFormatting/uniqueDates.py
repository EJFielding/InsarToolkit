#!/usr/bin/env python3
"""
    List the unique dates for a list of files
"""

### IMPORT MODULES ---
from glob import glob
from dateFormatting import udatesFromPairs


### PARSER ---
def createParser():
    import argparse
    parser = argparse.ArgumentParser(description='List unique dates from list of date pairs.')
    # Search for files in folder
    parser.add_argument('-f','--filesearch', dest='fsearch', type=str, default=None, help='Directory or search string.')
    parser.add_argument('-t','--ftype', dest='ftype', type=str, default=None, help='File type (ARIA, MintPy)')
    # Directly provide list of pairs
    parser.add_argument('-l','--list', dest='nameList', type=str, default=None, help='Directly specify list of pairs.')
    # Outputs
    parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode.')
    parser.add_argument('-o','--outname', dest='outName', type=str, default=None, help='Filename to write all unique dates.')
    return parser

def cmdParser(inpt_args=None):
    parser = createParser()
    return parser.parse_args(inpt_args)



### ANCILLARY FUNCTIONS ---
## Detect file type
def detectFtype(inpt):
    # File type
    if inpt.names[0].split('.')[-1]=='.nc':
        inpt.ftype='ARIA'
    elif inpt.names[0].split('.')[-1]=='.h5':
        inpt.ftype='MintPy'
    elif inpt.names[0].split('.')[-1]=='csv'
        inpt.ftype='csv'
    else:
        print('Could not detect file type; please specify manually using the -t option.')
        exit()

    # Report if requested
    if inpt.verbose is True: print('Filetype: {}'.format(inpt.ftype))


## Pair names from ARIA files
def ARIApairs(inpt):
    from nameFormatting import ARIAname
    ariaNames=[ARIAname(fname) for fname in inpt.names]
    pairs=[fname.dates for fname in ariaNames]

    return pairs


## Pair names from MintPy data set
def MintPyPairs(inpt):
    import h5py
    from dateFormatting import formatHDFdates
    with h5py.File(inpt.fsearch,'r') as F:
        dates,pairs=formatHDFdates(F['date'])
        F.close()

    return pairs


## Format list of pairs from YYYYMMDD_YYYYMMDD form to ['YYYYMMDD','YYYYMMDD']
def formatPairs(inpt):
    print(inpt.nameList)


## Read data from ASF csv
def dateFromCSV(inpt):
    import pandas as pd



### MAIN ---
if __name__=="__main__":
    # Gather arguments
    inpt=cmdParser()


    ## Load list of pairs
    # Search for files
    if inpt.fsearch:
        inpt.names=glob(inpt.fsearch)

        # Detect file type if not specified manually
        if not inpt.ftype:
            detectFtype(inpt)


        ## Grab filenames
        if inpt.ftype.lower()=='aria':
            pairs=ARIApairs(inpt)
        elif inpt.ftype.lower()=='mintpy':
            if inpt.fsearch.find('timeseries')!=-1:
                dates=MintPyTimeseriesDates(inpt)
            pairs=MintPyPairs(inpt)


    ## Input files as list
    if inpt.nameList:
        pairs=formatPairs(inpt.nameList)


    ## Unique dates from pairs
    dates=udatesFromPairs(pairs)
    nDates=len(dates)

    if inpt.verbose is True:
        print('Unique dates:')
        [print(date) for date in dates]
        print('{} dates detected'.format(nDates))


    ## Save if requested
    if inpt.outName:
        if inpt.outName[-4:]!='.txt': inpt.outName+='.txt'
        with open(inpt.outName,'w') as outFile:
            [outFile.write('{}\n'.format(date)) for date in dates]
            outFile.close()

        if inpt.verbose is True: print('Saved to: {}'.format(inpt.outName))
