import ROOT
import numpy as np
import root_numpy as rnp
from scipy import spatial


def lattice_doms(detfile):
    from detector_positions import structured_positions
    doms, pmts = structured_positions(detfile)

    dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])

    # lattice definition (larger than detector)
    xyz = []
    for x in range(-300, 1100, 90):
        for y in np.arange(-550, 550, 45 * np.sqrt(3)):
            for i, z in enumerate(range(98, 712, 36)):
                xyz.append((x, y, z))
            x -= 45
    xyz = np.asarray(xyz)

    lattice_doms = []
    for pt in doms:
        x, y, z = pt["x"], pt["y"], pt["z"]
        lattice_doms.append((xyz[spatial.KDTree(xyz).query((x, y, z))[1]][0],
                             xyz[spatial.KDTree(xyz).query((x, y, z))[1]][1],
                             xyz[spatial.KDTree(xyz).query((x, y, z))[1]][2]))

    lattice_doms_arr = np.asarray(lattice_doms, dtype=dt)

    xyz = []
    for x in range(-300, 1100, 90):
        for y in np.arange(-550, 550, 45 * np.sqrt(3)):
            for i, z in enumerate(range(98, 712, 36)):
                xyz.append((x, y, z))
            x -= 45
    xyz = np.asarray(xyz, dtype=dt)

    return xyz, lattice_doms_arr
