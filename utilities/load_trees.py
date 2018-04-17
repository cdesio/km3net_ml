
import numpy as np
import root_numpy as rnp


def load_trees(filename):
    """
    Function to import the trees from the aanet input file

    Parameters:
    -----------
    filename : str
        aanet input file
    Returns:
    --------
    ch_id : np.ndarray
        array containing the id of the pmts hit for all of the events (triggered and not triggered) 
    dom_id : np.ndarray
        array containing the id of the doms hit for all of the events (triggered and not triggered) 
    trig : np.ndarray
        array containing the flag `0` or `1` indicating whether the event has been triggered. 
        The information is stored for each hit
    t : np.ndarray
        array containing the times of the hits for all of the triggered events 
    
    """
    
    ch_id = rnp.root2array(filename, treename="E", branches="Evt.hits.channel_id")
    dom_id = rnp.root2array(filename, treename="E", branches="Evt.hits.dom_id")
    trig = rnp.root2array(filename, treename= "E", branches="Evt.hits.trig")
    t = rnp.root2array(filename, treename="E", branches="Evt.hits.t")
    times = np.asarray([t[evt][trig[evt]==True] for evt in range(t.size)])
    
    return ch_id, dom_id, trig, times




def load_reco_files(filename):
    """
    Function to import the trees from the aanet input file

    Parameters:
    -----------
    filename : str
        aanet input file

    """

    dir_x = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__dx")
    dir_y = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__dy")
    dir_z = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__dz")
    pos_x = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__x")
    pos_y = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__y")
    pos_z = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__z")
    Energy = rnp.root2array(filename, treename="EVT", branches="vector<JFIT::JFit>.__E")


    return dir_x, dir_y, dir_z, pos_x, pos_y, pos_z, Energy