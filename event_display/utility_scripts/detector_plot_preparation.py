import ROOT
import numpy as np
import root_numpy as rnp

NFLOORS = 18
NSTRING = 115
PMTSPERDOM = 31
pmtstot = NFLOORS * NSTRING * PMTSPERDOM
ndoms = NFLOORS * NSTRING
coord_origin = np.asarray((13.887, 6.713, 405.932))


def plot_simulated_evts(fname, detfile, flag):
    from detector_positions import structured_positions
    doms, pmts = structured_positions(detfile)

    from load_trees import load_trees
    ch_id, dom_id, trig, times = load_trees(fname)

    if (not ch_id.size or not dom_id.size or not trig.size or not times.size):
        print("load trees...2")
        ch_id, dom_id, trig, times = load_trees(fname)
    else:
        print("trees correctly loaded")

    def hits_positions(evt, dom_id, trig, ch_id):
        """
        Function to calculate the 3D positions of the triggered events

        Parameters:
        -----------
        evt : np.int
            the event id
        Returns:
        --------
        ppmts_hit : np.ndarray
            array containing the positions of the pmts hit for the selected event 
        pdoms_hit : np.ndarray
            array containing the positions of the doms hit for the selected event  
        """
        dom_filter = dom_id[evt][trig[evt] == True] - 1
        pmt_filter = (dom_filter * PMTSPERDOM) + ch_id[evt][trig[evt] == True]
        ppmts_hit = pmts[pmt_filter]
        pdoms_hit = doms[dom_filter]
        return ppmts_hit, pdoms_hit

    pmts_hit = []
    doms_hit = []
    for evt in range(0, dom_id.size):
        pm, dm = hits_positions(evt, dom_id, trig, ch_id)
        pmts_hit.append(pm)
        doms_hit.append(dm)

    pmts_hit = np.asarray(pmts_hit)
    doms_hit = np.asarray(doms_hit)

    from mc_positions import simulated_particle_positions
    if (flag == "numu"):
        mc_positions = simulated_particle_positions(fname, 5)
    elif (flag == "nue"):
        mc_positions = simulated_particle_positions(fname, 3)
    elif (flag == "muatm"):
        mc_positions = simulated_particle_positions(fname, -13)

    # times normalization

    norm_times = []
    for i, evt in enumerate(times):
        norm_times.append(
            np.asarray([(evt[j] - (np.min(evt))) / (np.max(evt) - np.min(evt)) for j in range(evt.shape[0])]))
    norm_times = np.asarray(norm_times)

    return doms_hit, norm_times, mc_positions
