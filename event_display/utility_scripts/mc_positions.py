import ROOT
import root_numpy as rnp
import numpy as np


NFLOORS = 18
NSTRING = 115
PMTSPERDOM = 31
pmtstot = NFLOORS * NSTRING * PMTSPERDOM
ndoms = NFLOORS * NSTRING
coord_origin = np.asarray((13.887,6.713,405.932))

def simulated_particle_positions(filename, pid):
     
    """
    Function to calculate the 
    The color scale from blue to red shows the hit times. Event starts at blue and ends in red

    Parameters:
    -----------
    evt : np.int
         event id to plot
    Returns:
    --------
    plot : matplotlib scatter
        scatter plot of the chosen numucc event  
    mc : mc_points 
        scatter plot of start and end position according to mc and line to display the mc track
    """
    #import positions and trk_type to select muons (`pid==5`) or electrons ( `pid==3`) 
    
    pos_x = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.pos.x") + coord_origin[0]
    pos_y = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.pos.y") + coord_origin[1]
    pos_z = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.pos.z") + coord_origin[2]
    trk_type = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.type")
    
    #define dtype for the array of records
    
    dt = np.dtype([('x', np.float64), ('y', np.float64), ('z', np.float64)])
    
    mc_pos = []
    #for each event, select the positions (x,y,z) of the particle/first-particle of the bundle
    for evt in range(pos_x.shape[0]):
        mc_pos.append((
            (np.float(pos_x[evt][trk_type[evt]==pid]) if pos_x[evt][trk_type[evt]==pid].size==1 else np.float(pos_x[evt][trk_type[evt]==pid][0]), 
             np.float(pos_y[evt][trk_type[evt]==pid]) if pos_y[evt][trk_type[evt]==pid].size==1 else np.float(pos_y[evt][trk_type[evt]==pid][0]), 
             np.float(pos_z[evt][trk_type[evt]==pid]) if pos_z[evt][trk_type[evt]==pid].size==1 else np.float(pos_z[evt][trk_type[evt]==pid][0])
            )))
    mc_start = np.asarray(mc_pos, dtype=dt)
    
    #import trk_len and directions to calculate the end point
    
    trk_len = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.len")
    
    dir_x = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.dir.x")
    dir_y = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.dir.y")
    dir_z = rnp.root2array(filename, treename="E", branches="Evt.mc_trks.dir.z")
    
    #select particle tracks(trk_type==5 || trk_type==3)
    trks_len = []
    for i in range(trk_len.shape[0]):
        length = np.float(trk_len[i][trk_type[i]==pid]) if trk_len[i][trk_type[i]==pid].size==1 else np.float(trk_len[i][trk_type[i]==pid][0])
        trks_len.append(length)
    trks_len = np.asarray(trks_len)
    
    #select particle directions(trk_type==5 || trk_type==3)
    mc_dir = []
    for evt in range(dir_x.shape[0]):
        mc_dir.append((
            (np.float(dir_x[evt][trk_type[evt]==pid]) if dir_x[evt][trk_type[evt]==pid].size==1 else np.float(dir_x[evt][trk_type[evt]==pid][0]), 
             np.float(dir_y[evt][trk_type[evt]==pid]) if dir_y[evt][trk_type[evt]==pid].size==1 else np.float(dir_y[evt][trk_type[evt]==pid][0]), 
             np.float(dir_z[evt][trk_type[evt]==pid]) if dir_z[evt][trk_type[evt]==pid].size==1 else np.float(dir_z[evt][trk_type[evt]==pid][0])
            )))
    mc_dir = np.asarray(mc_dir, dtype=dt)
    
    #calculate the end point of the simulated track
    mc_end = []
    for evt in range(mc_start.shape[0]):
        mc_end.append((
            (mc_start[evt]['x'] + mc_dir[evt]['x']*np.abs(trks_len[evt]),
             mc_start[evt]['y'] + mc_dir[evt]['y']*np.abs(trks_len[evt]),
             mc_start[evt]['z'] + mc_dir[evt]['z']*np.abs(trks_len[evt])) if trks_len[evt] else
            (mc_start[evt]['x'] + mc_dir[evt]['x']* 100,
             mc_start[evt]['y'] + mc_dir[evt]['y']* 100,
             mc_start[evt]['z'] + mc_dir[evt]['z']* 100)))
    mc_end = np.asarray(mc_end, dtype=dt)
    
    return mc_start, mc_end