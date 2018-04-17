import numpy as np


def doms_positions(filename):
    NFLOORS = 18
    NSTRING = 115
    PMTSPERDOM = 31
    pmtstot = NFLOORS * NSTRING * PMTSPERDOM

    pmts_pos = []
    with open(filename) as f:
        for line in f:
            if line.startswith(' '):
                pmts_pos.append(((np.float(line.split()[1]), np.float(line.split()[2]), np.float(line.split()[3]))))
    pos = np.asarray(pmts_pos)
    dom_pos = [(np.mean(pos[i:i + 31][:, 0]), np.mean(pos[i:i + 31][:, 1]), np.mean(pos[i:i + 31][:, 2])) for i in
               range(0, pmtstot, PMTSPERDOM)]
    doms = np.asarray(dom_pos)
    return doms, pos


def structured_positions(filename):
    NFLOORS = 18
    NSTRING = 115
    PMTSPERDOM = 31
    pmtstot = NFLOORS * NSTRING * PMTSPERDOM
    dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])
    pmts_pos = []
    with open(filename) as f:
        for line in f:
            if line.startswith(' '):
                pmts_pos.append(((np.float(line.split()[1]), np.float(line.split()[2]), np.float(line.split()[3]))))
    pos = np.asarray(pmts_pos, dtype=dt)
    dom_pos = [(np.mean(pos[i:i + 31]['x']), np.mean(pos[i:i + 31]['y']), np.mean(pos[i:i + 31]['z'])) for i in
               range(0, pmtstot, PMTSPERDOM)]
    doms = np.asarray(dom_pos, dtype=dt)
    return doms, pos
