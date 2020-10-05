#!/usr/bin/env python3



### IMPORT MODUELS ---
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from Templates import prepareTemplate
from DecompositionModules import decomposeISCE



### PARSER ---
Description='''Decompose two or more overlapping signals (interferograms, LOS velocity maps) 
into EW and vertical components. This method is based on Wright et al., 2004 and 
assumes zero sensitivity to motion in the NS direction.

This function is (or will be) designed to work with:
- ISCE interferograms
- ARIA standard products
- ARIA stitched products
- MintPy velocity maps

To generate the input template...
'''

def createParser():
    parser = argparse.ArgumentParser(description = Description,
        formatter_class = argparse.RawTextHelpFormatter)

    # Create template
    templateArgs = parser.add_argument_group('TEMPLATE ARGUMENTS')
    templateArgs.add_argument('-t', '--template', dest='template', type=str, default=None,
        help='Generate a YAML file with the necessary items and dummy values')
    templateArgs.add_argument('--n-datasets', dest='nDatasets', type=int, default=2,
        help='Number of input data sets for template')

    # Input file already created
    inputArgs = parser.add_argument_group('INPUT ARGUMENTS')
    inputArgs.add_argument('-f', '--fname', dest='fname', type=str, default=None,
        help='YAML file with input data sets and parameters')

    # Outputs
    outputArgs = parser.add_argument_group('OUTPUT ARGUMENTS')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
        help='Plot inputs and outputs')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


### ANCILLARY FUNCTIONS ---
## Confirm output directory exists
def confirmOutputDir(outFolder):
    '''
    Confirm the existence of the output directory.
    If it does not exist, create it.
    '''
    # Convert outName to aboslute path
    outFolder = os.path.abspath(outFolder)

    # Create directory if it does not exist
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)


## Read input yaml file
def parseInputs(fname):
    '''
    Read input yaml.
    '''
    with open(fname,'r') as fIn:
        # Load yaml content
        docs = yaml.load_all(fIn, Loader=yaml.FullLoader)

        # Split into formatting info, and data sets
        Datasets = []
        for n,doc in enumerate(docs):
            if n == 0: 
                Fmt = doc
            else:
                Datasets.append(doc)

        fIn.close()

    # Check formatting
    checkFormatting(Fmt)
    print('*'*31)

    # Check that data sets provide necessary parameters
    print('{:d} data sets discovered'.format(len(Datasets)))
    for dataset in Datasets:
        checkDataset(dataset)
    print('*'*31)

    # Return results
    return Fmt, Datasets

# Check input formatting from yaml file
def checkFormatting(Fmt):
    '''
    Check input formatting to avoid future headaches.
    '''
    # Report project name
    print('Project name: {:s}'.format(Fmt['projectName']))

    # Check data type
    print('Data type: {:s}'.format(Fmt['format']))
    if Fmt['format'] not in ['ISCE']:
        print('Format type {} not valid or not ready yet!'.\
            format(Fmt['format']))
        exit()

# Check data set inputs
def checkDataset(dataset):
    '''
    Check that the inputs for a dataset satisfy that data set's needs.
    '''
    # Identify data set
    print('{:s}'.format(dataset['name']))

    # Check that all necessary parameters are filled in properly
    print('... Good.')



### MAIN ---
if __name__ == '__main__':
    ## Parse and check inputs
    # Gather inputs
    inps = cmdParser()

    # Input file or request for template
    if inps.template is not None and inps.fname is not None:
        print('\nError: Specify only template or fname; not both\n')
        exit()


    ## Prepare template, or ...
    if inps.template:
        print('Preparing template...')
        prepareTemplate(inps.template,nDatasets=inps.nDatasets)


    ## ... Read prepared file
    elif inps.fname:
        print('Reading from file: {:s}'.format(inps.fname))

        # Unpack yaml file
        Fmt, Datasets = parseInputs(inps.fname)

        # Prepare output directory if need be
        confirmOutputDir(Fmt['outFolder'])

        # Decompose signals based on data types
        if Fmt['format'] == 'ISCE': 
            decomposeISCE(Fmt,Datasets,plot=inps.plot)


    ## Don't know what to do
    else:
        print('\nError: Specify template (-t) or input file (-f)\n')


    ## Plot results if requested
    if inps.plot: plt.show()

