from os import path as p
import ROOT
from ROOT import gSystem
import root_numpy as rnp
import numpy as np
import os
import pickle
from dependencies import root_dependencies
root_dependencies()


def trigger_map(filename):
    montecarlo_eventnumber = rnp.root2array(filename, treename="MONTECARLO", branches="eventNumber_")
    triggered_counter = rnp.root2array(filename, treename="KM3NET_EVENT", branches="KM3NET_EVENT.getCounter()")
    triggered_counter = triggered_counter.astype(np.int64)
    triggered_maps = np.zeros(montecarlo_eventnumber.shape, dtype=np.bool)
    triggered_maps[triggered_counter] = True
    return triggered_maps
		
