from os import path as p
import ROOT
from ROOT import gSystem
import root_numpy as rnp
import numpy as np
import os
import pickle

def root_dependencies():

	JPP_LIB = os.environ['JPP_LIB']
	JPP_DIR = os.environ['JPP_DIR']
	LIBLANG = p.join(JPP_LIB, "liblang.so")
	LIBTRIGGER = p.join(JPP_LIB, "libtriggerROOT.so")
	LIBKM3NETDAQ = p.join(JPP_LIB, "libKM3NeTDAQROOT.so")
	LIBANTCC = p.join(JPP_DIR, "externals/antares-daq/out/Linux/lib/libantccROOT.so")

	gSystem.Load(LIBLANG)
	gSystem.Load(LIBTRIGGER)
	gSystem.Load(LIBKM3NETDAQ)
	gSystem.Load(LIBANTCC)