import numpy as np


def tslices(times_numu, times_nue):
    min_t_nue = np.min(np.hstack(times_nue))
    max_t_nue = np.max(np.hstack(times_nue))
    min_t_numu = np.min(np.hstack(times_numu))
    max_t_numu= np.max(np.hstack(times_numu))
    t_range_min = np.min((min_t_nue, min_t_numu))
    t_range_max = np.max((max_t_nue, max_t_numu))
    #timeslices = np.arange(t_range_min, t_range_max+200, 150)
    timeslices = np.linspace(t_range_min, t_range_max+200, 76)
    print(t_range_min, t_range_max)
    return timeslices


def tslices_multi_files(times_numu1, times_nue1, times_numu2, times_nue2, times_numu3, times_nue3):
    min_t_nue1 = np.min(np.hstack(times_nue1))
    max_t_nue1 = np.max(np.hstack(times_nue1))
    min_t_numu1 = np.min(np.hstack(times_numu1))
    max_t_numu1 = np.max(np.hstack(times_numu1))
                        
    min_t_nue2 = np.min(np.hstack(times_nue2))
    max_t_nue2 = np.max(np.hstack(times_nue2))
    min_t_numu2 = np.min(np.hstack(times_numu2))
    max_t_numu2 = np.max(np.hstack(times_numu2))
                        
    min_t_nue3 = np.min(np.hstack(times_nue3))
    max_t_nue3 = np.max(np.hstack(times_nue3))
    min_t_numu3 = np.min(np.hstack(times_numu3))
    max_t_numu3 = np.max(np.hstack(times_numu3))
    
    t_range_min = np.min((min_t_nue1, min_t_numu1, min_t_nue2, min_t_numu2, min_t_nue3, min_t_numu3 ))
    t_range_max = np.max((max_t_nue1, max_t_numu1, max_t_nue2, max_t_numu2, max_t_nue3, max_t_numu3))
    timeslices = np.arange(t_range_min, t_range_max+200, 150)
    print(t_range_min, t_range_max)
    return timeslices