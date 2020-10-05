### IMPORT MODULES ---
import numpy as np


### ISCE ---
def ISCE_look_to_vector(LOSds):
    '''
    Convert incidence angle and azimuth to 3D target-to-satellite look vectors.
    This follows this ISCE convention: http://earthdef.caltech.edu/boards/4/topics/1915

    From the los.rdr.geo.xml file:
    Two channel Line-Of-Sight geometry image (all angles in degrees).
    Represents vector drawn from target to platform.
    Channel 1: Incidence angle measured from vertical at target (always +ve).
    Channel 2: Azimuth angle measured from North in Anti-clockwise direction.
    '''

    # Parse LOS data set
    inc = LOSds.GetRasterBand(1).ReadAsArray()
    az = LOSds.GetRasterBand(2).ReadAsArray()

    # Mask arrays
    inc = np.ma.array(inc,mask=(inc==0.))
    az = np.ma.array(az,mask=(az==0.))

    # Inputs to radians
    #  Add pi/2 to az because az is measured w.r.t. N
    inc = np.pi/180*inc
    az = np.pi/2+np.pi/180*az

    # Convert horizontal angle to vector
    #  Scale by sin(inc) because inc is w.r.t. vertical
    Lx = np.cos(az)*np.sin(inc)
    Ly = np.sin(az)*np.sin(inc)

    # Convert vertical angle to vector
    #  Use cos because inc is w.r.t. vertical
    Lz = np.cos(inc)

    # Scale to unit length
    L = (Lx**2 + Ly**2 + Lz**2)**0.5

    Lx/=L
    Ly/=L
    Lz/=L

    # Report
    print('Converted LOS to look vectors using ISCE convention:',
        'Inc wrt vertical; Az CCW wrt north')
    print('Averages: Lx {:.4f}; Ly {:.4f}; Lz {:.4f}'. \
        format(Lx.mean(),Ly.mean(),Lz.mean()))

    return Lx, Ly, Lz