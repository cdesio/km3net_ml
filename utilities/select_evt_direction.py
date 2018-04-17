import ROOT
import numpy as np
import root_numpy as rnp


MUATM_PART_CODE = -13
NUE_PART_CODE = 3
NUMU_PART_CODE = 5


def sel_up_down(fname, part_code):
    """

    Parameters
    ----------
    fname
    part_code

    Returns
    -------

    """

    trk_type = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.type")
    trk_dir_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.x")
    trk_dir_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.y")
    trk_dir_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.z")
    print(" reload")  # FIXME: Reload to make sure that r2np is working properly!
    trk_type = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.type")
    trk_dir_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.x")
    trk_dir_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.y")
    trk_dir_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.z")

    def sel_trk(trk_dir, trk_type, part_code):
        return (np.asarray([trk_dir[evt][trk_type[evt] == part_code][0] for evt in range(trk_type.shape[0])]))


    sel_trk_dir_z = sel_trk(trk_dir_z, trk_type, part_code)

    upgoing_indx = np.where(sel_trk_dir_z >= 0.25)[0]
    downgoing_indx = np.where(sel_trk_dir_z <= -0.25)[0]
    return upgoing_indx, downgoing_indx


def sel_up_down_5doms(fname, part_code, selection):
    """

    Parameters
    ----------
    fname
    part_code
    selection

    Returns
    -------

    """

    trk_type = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.type")
    trk_dir_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.x")
    trk_dir_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.y")
    trk_dir_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.z")

    trk_pos_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.x")
    trk_pos_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.y")
    trk_pos_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.z")

    energy = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.E")
    print(" reload")

    trk_type = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.type")

    trk_dir_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.x")
    trk_dir_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.y")
    trk_dir_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.dir.z")

    trk_pos_x = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.x")
    trk_pos_y = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.y")
    trk_pos_z = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.pos.z")

    energy = rnp.root2array(fname, treename="E", branches="Evt.mc_trks.E")

    def sel_trk_5doms(trk_arr, trk_type, part_code):
        trk_sel = trk_arr[selection]
        trk_type_sel = trk_type[selection]

        sel_arr = np.asarray(
            [trk_sel[evt][trk_type_sel[evt] == part_code][0] for evt in range(trk_type_sel.shape[0])])
        return sel_arr


    sel_trk_dir_x = sel_trk_5doms(trk_dir_x, trk_type, part_code)
    sel_trk_dir_y = sel_trk_5doms(trk_dir_y, trk_type, part_code)
    sel_trk_dir_z = sel_trk_5doms(trk_dir_z, trk_type, part_code)

    sel_trk_pos_x = sel_trk_5doms(trk_pos_x, trk_type, part_code)
    sel_trk_pos_y = sel_trk_5doms(trk_pos_y, trk_type, part_code)
    sel_trk_pos_z = sel_trk_5doms(trk_pos_z, trk_type, part_code)

    sel_energy = sel_trk_5doms(energy, trk_type, part_code)

    upgoing_indx = np.where(sel_trk_dir_z >= 0)[0]
    downgoing_indx = np.where(sel_trk_dir_z < 0)[0]

    return upgoing_indx, downgoing_indx, sel_trk_dir_x, sel_trk_dir_y, sel_trk_dir_z, \
           sel_trk_pos_x, sel_trk_pos_y, sel_trk_pos_z, sel_energy
