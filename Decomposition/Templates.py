### TEMPLATES ---
## ISCE
ISCE_formatstr = '''---
# Formatting
format: ISCE
projectName: xxx
outFolder: ./

'''

ISCE_datasetstr = '''---
# Data set {0:d}
name: xxx
folder: xxx

'''


### FUNCTION ---
def prepareTemplate(templateType,nDatasets=2):
    '''
    This function prepares one of the templates above for use.
    '''
    print('Type: {:s}'.format(templateType))
    print('{:d} data sets'.format(nDatasets))

    outName = '{:s}_inputs.yaml'.format(templateType)

    with open(outName,'w') as outFile:
        outFile.write(ISCE_formatstr)
        for n in range(nDatasets):
            outFile.write(ISCE_datasetstr.format(n+1))
        outFile.close()

    print('Template saved to >\n{:s}'.format(outName))