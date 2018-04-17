import numpy as np
from scipy import spatial

detfile = "utilities/km3net_jul13_90m.detx"

from lattice_doms_znewk import lattice_doms

lattice, l_doms = lattice_doms(detfile)

np.save("utilities/lattice.npy", lattice)
np.save("utilities/l_doms.npy", l_doms)
