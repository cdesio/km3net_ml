
from scipy.spatial.distance import euclidean
import numpy as np

PC = (13.887,6.713,405.932)


def distance_neutrino_detector_centre(P0, direction):
    """

    Parameters
    ----------
    P0 : tuple
        coordinates of the incoming particle
    PC : tuple
        coordinates of the detector centre
    dx : float
        x direction cosine of the particle
    dy : float
        y direction cosine of the particle
    dz : float
        z direction cosine of the particle

    Returns
    -------
    distance : float
        computed euclidean distance between the incoming particle direction and the detector centre

    Examples
    --------

    >>> dir = (0.317429, -0.2461639, 0.9157739)
    >>> P0 = (-398.349,  304.152,  54.845)
    >>> d = distance_neutrino_detector_centre(P0, dir)
    """
    x0, y0, z0 = P0
    xc, yc, zc = PC
    dx, dy, dz = direction

    A = dx * xc
    B = dy * yc
    C = dz * zc
    D = dx * x0
    E = dy * y0
    F = ((1 / dz) + dz) * z0

    z = (A + B + C - D - E - F) * dz
    x = x0 + (dx / dz) * (z - z0)
    y = y0 + (dy / dz) * (z - z0)

    return euclidean((xc, yc, zc), (x, y, z)), (x,y,z)
    

def distance_cross_product(P0, direction):
    """

    Parameters
    ----------
    P0 : tuple
        coordinates of the incoming particle
    PC : tuple
        coordinates of the detector centre
    dx : float
        x direction cosine of the particle
    dy : float
        y direction cosine of the particle
    dz : float
        z direction cosine of the particle

    Returns
    -------
    distance : float
        computed euclidean distance between the incoming particle direction and the detector centre

    Examples
    --------

    >>> dir = (0.317429, -0.2461639, 0.9157739)
    >>> P0 = (-398.349,  304.152,  54.845)
    >>> d = distance_neutrino_detector_centre(P0, dir)
    """
    x0, y0, z0 = P0
    xc, yc, zc = PC
    dx, dy, dz = direction
    
    P0 = np.asarray(P0)
    PC= np.asarray(PC)
    distance_vector = P0-PC
    cross_vector = np.cross(distance_vector, direction)
    
    distance = np.sqrt(np.sum(cross_vector**2))
    
    return distance    

def distance_calculation(Xy_distance):
   
    pos_x = Xy_distance["posx"]
    pos_y = Xy_distance["posy"]
    pos_z = Xy_distance["posz"]
    dir_x = Xy_distance["dirx"]
    dir_y = Xy_distance["diry"]
    dir_z = Xy_distance["dirz"]
    distances = []
    for evt in range(pos_x.shape[0]):
        P0 = (pos_x[evt],pos_y[evt],pos_z[evt])
    
        direct = (dir_x[evt],dir_y[evt],dir_z[evt])

        transl_P0 = [p+c for p,c in zip(P0,PC)] 
    
        dist = distance_cross_product(transl_P0, direct)
        distances.append(dist)
        print("distance: {}".format(dist))
    return np.asarray(distances)
