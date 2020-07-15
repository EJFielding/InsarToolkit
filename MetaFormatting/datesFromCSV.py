#!/usr/bin/env python3
"""
    Given a metadata csv file from ASF Vertex, list all the
     unique dates in that file.
"""

### IMPORT MODULES ---
import numpy as np
import pandas as pd
import argparse


### PARSER ---
def createParser():
    parser = argparse.ArgumentParser( description='List unique dates from csv file')
    parser.add_argument(dest='csvfile', type=str, help='Full path to CSV file containing date list.')
    parser.add_argument('--column-name', dest='columnName', type=str, default='Acquisition Date', help='Column header for date list')

    parser.add_argument('-v','--verbose', dest='verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-o','--outName', dest='outName', type=str, default=None, help='Output name')
    return parser

def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### MAIN ---
def main(iargs=None):
    # Gather inputs
    inps = cmdLineParse(iargs)

    # Open file
    data=pd.read_csv(inps.csvfile)

    # Grab dates
    datetimes=data.loc[:,inps.columnName]
    dates=[datetime.split('T')[0] for datetime in datetimes]

    # Ensure only unique dates are included
    dates=set(dates)
    dates=list(dates)
    dates.sort()

    # Format into YYYYMMDD
    dates=[date.replace('-','') for date in dates]

    nDates=len(dates)

    # Report if requested
    if inps.verbose==True:
        print('Loaded: {}'.format(inps.csvfile))
        print('Columns: {}'.format(data.columns))
        print('Dates: {}'.format(dates))
        print('Nb unique dates: {}'.format(nDates))

    # Write to file if requested
    if inps.outName:
        outName=inps.outName+'.txt'
        with open(outName,'w') as outFile:
            [outFile.write(date+'\n') for date in dates]
            outFile.close()
        if inps.verbose==True: print('Saved to: {}'.format(outName))


if __name__=='__main__':
    main()