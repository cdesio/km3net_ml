#!/usr/bin/python
import argparse
import os

import ROOT
import numpy as np
import root_numpy as rnp
from load_trees import load_trees

from timeslices import tslices
from select_evt_direction import sel_up_down_5doms
from select_evt_direction import NUE_PART_CODE, NUMU_PART_CODE, MUATM_PART_CODE
from itertools import product



def sel_doms_hit(ndoms, dom_id, times, trig):
    """

    Parameters
    ----------
    ndoms
    dom_id
    times
    trig

    Returns
    -------

    """
    trig_evts = np.asarray([dom_id[evt][trig[evt] == True] for evt in range(dom_id.shape[0])])
    unique_doms_hit = np.asarray([np.unique(trig_evts[evt],
                                            return_index=True)[0] for evt in range(trig_evts.shape[0])])

    unique_doms_indx = np.asarray([np.unique(dom_id[evt][trig[evt] == True],
                                             return_index=True)[1] for evt in range(dom_id.shape[0])])

    selection = np.asarray([unique_doms_hit[evt].size >= ndoms for evt in range(unique_doms_hit.shape[0])])
    sel_n_doms = unique_doms_hit[selection]
    sel_times = np.asarray([times[evt][unique_doms_indx[evt]] for evt in range(times.shape[0])])[selection]
    return sel_n_doms, sel_times, selection


def Xy_creation(dom_id, tslice, times, flag, doms_map, lol):
    """

    Parameters
    ----------
    doms_map
    dom_id
    tslice
    times
    flag

    Returns
    -------

    """

    nu_events = dom_id.shape[0]
    n_timeslices = tslice.shape[0] - 1
    lattice_shape = (16, 15, 18)
    X_nu = np.zeros((nu_events, n_timeslices,) + lattice_shape)

    for evt in range(nu_events):
        scaled_dom_id = dom_id[evt] - 1
        times_event_hits = times[evt]
        for ts, tsl in enumerate(zip(tslice[:-1], tslice[1:])):
            low, high = tsl
            hits = np.where((times_event_hits >= low) & (times_event_hits < high))[0]
            if not len(hits):
                continue
            dom_hit_in_slice = scaled_dom_id[hits]
            l_dom_hit_in_slice = doms_map[dom_hit_in_slice]
            l_ret = lol[l_dom_hit_in_slice]
            for dom_indx in l_ret:
                X_nu[evt, ts, dom_indx[0], dom_indx[1], dom_indx[2]] += 1

    if (flag == 'nu_muon'):
        Y_nu = np.ones(dom_id.shape[0])
    elif (flag == 'electron'):
        Y_nu = np.zeros(dom_id.shape[0])
    elif (flag == 'muon'):
        Y_nu = 2 * np.ones(dom_id.shape[0])

    return X_nu, Y_nu


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script creates Xy data from aa.root data files')

    parser.add_argument('--inumu', help='numu input file name', required=True)
    parser.add_argument('--inue', help='nue input file name', required=True)
    parser.add_argument('--onumu', help='numu output file name', required=True)
    parser.add_argument('--onue', help='nue output file name', required=True)
    parser.add_argument("--outdir", help="output directory", required=True)
    args = parser.parse_args()

    print("Input files: %s %s" % (args.inumu, args.inue))
    print("Output file: %s %s" % (args.onumu, args.onue))

    detfile = "utilities/km3net_jul13_90m.detx"
    nuefile = args.inue
    numufile = args.inumu

    print("load trees...1")

    ch_id_numu, dom_id_numu, trig_numu, times_numu = load_trees(numufile)
    ch_id_nue, dom_id_nue, trig_nue, times_nue = load_trees(nuefile)

    if (not ch_id_numu.size or not dom_id_numu.size or not trig_numu.size or not times_numu.size or
            not ch_id_nue.size or not dom_id_nue.size or not trig_nue.size or not times_nue.size):
        print("load trees...2")
        # repeat - workaround

        ch_id_numu, dom_id_numu, trig_numu, times_numu = load_trees(numufile)
        ch_id_nue, dom_id_nue, trig_nue, times_nue = load_trees(nuefile)
    else:
        print("trees correctly loaded")

    print("load lattice")
    # from lattice_doms_znewk import lattice_doms
    # lattice, l_doms = lattice_doms(detfile)
    lattice = np.load("utilities/lattice.npy")
    l_doms = np.load("utilities/l_doms.npy")

    isin_lattice = np.isin(lattice, l_doms)
    lattice_idx = np.argwhere(isin_lattice).flatten()
    l_doms_idx = np.argsort(np.array([np.argwhere(l_doms == a) for a in lattice[isin_lattice]]).flatten())
    doms_map = lattice_idx[l_doms_idx]

    ii, jj, kk = range(16), range(15), range(18)
    lol = np.asarray(list(product(ii, jj, kk)))

    print("Selecting events with more than 5 doms hit")

    sel_5doms_numu, sel_times_numu, ids_numu = sel_doms_hit(5, dom_id_numu, times_numu, trig_numu)
    sel_5doms_nue, sel_times_nue, ids_nue = sel_doms_hit(5, dom_id_nue, times_nue, trig_nue)

    tslice = tslices(sel_times_numu, sel_times_nue)

    print("Xy creation")

    X_numu, Y_numu = Xy_creation(sel_5doms_numu, tslice, sel_times_numu, 'nu_muon', doms_map, lol)
    X_nue, Y_nue = Xy_creation(sel_5doms_nue, tslice, sel_times_nue, 'electron', doms_map, lol)

    upgoing_numu_indx, downgoing_numu_indx, \
    trk_numu_dir_x, trk_numu_dir_y, trk_numu_dir_z, \
    trk_numu_pos_x, trk_numu_pos_y, trk_numu_pos_z, \
    numu_energy = sel_up_down_5doms(numufile, NUMU_PART_CODE, ids_numu)

    upgoing_nue_indx, downgoing_nue_indx, \
    trk_nue_dir_x, trk_nue_dir_y, trk_nue_dir_z, \
    trk_nue_pos_x, trk_nue_pos_y, trk_nue_pos_z, \
    nue_energy = sel_up_down_5doms(nuefile, NUE_PART_CODE, ids_nue)

    # np.savez(args.outdir+args.onumu+"_sel5_dir_z"+".npz",z=trk_numu_dir_z)
    # np.savez(args.outdir+args.onue+"_sel5_dir_z"+".npz",z=trk_nue_dir_z)

    np.savez(os.path.join(args.outdir, "{}_sel5_doms_map.npz".format(args.onumu)), id=ids_numu)
    np.savez(os.path.join(args.outdir, "{}_sel_5_doms_map.npz".format(args.onue)), id=ids_nue)

    np.savez_compressed(os.path.join(args.outdir, "Xy_{}_sel5_doms.npz".format(args.onumu)),
                        x=X_numu.astype(np.uint8), y=Y_numu.astype(np.uint8),
                        dirx=trk_numu_dir_x, diry=trk_numu_dir_y,
                        dirz=trk_numu_dir_z, posx=trk_numu_pos_x,
                        posy=trk_numu_pos_y, posz=trk_numu_pos_z,
                        E=numu_energy)

    np.savez_compressed(os.path.join(args.outdir, "Xy_{}_sel5_doms.npz".format(args.onue)),
                        x=X_nue.astype(np.uint8), y=Y_nue.astype(np.uint8),
                        dirx=trk_nue_dir_x, diry=trk_nue_dir_y,
                        dirz=trk_nue_dir_z, posx=trk_nue_pos_x,
                        posy=trk_nue_pos_y, posz=trk_nue_pos_z,
                        E=nue_energy)

    np.savez(os.path.join(args.outdir, "{}_sel5_updown_map.npz".format(args.onumu)),
             up=upgoing_numu_indx, down=downgoing_numu_indx)
    np.savez(os.path.join(args.outdir, "{}_sel5_updown_map.npz".format(args.onue)),
             up=upgoing_nue_indx, down=downgoing_nue_indx)
