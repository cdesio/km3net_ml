import ROOT
from ROOT import gSystem
import root_numpy as rnp
import numpy as np
import os
import pickle


# Load file function
#The function is used to load the JTE processed files

def load_root_files(files_key, path_to_neutrino_files, st, nocomb, opt, optnew):
    """
    Loops on filenames in the path di and saves the filenames in lists according to the selected filters.
    
    Parameters:
    -----------
    files_key : str
        the file type, as 'numuCC', 'numuNC', etc...
    path_to_neutrino_files : str
        the path to the 'neutrino' directory
    st : str
        the extention of the "standard trigger" files
    nocomb : str
        the extention of the "nocombine" files
    opt : str
        the extention of the "optimised" files
    optnew : str
        the extention of the "optimised new" files


    Returns:
    --------
    st_files : list
        list of the paths of each "st" file
    nocomb_files : list
        list of the paths of each "nocomb" file
    opt_files : list
        list of the paths of each "opt" file
    optnew_files : list
        list of the paths of each "optnew" file
    
    """
    
    #definition of the output lists
    st_files = []
    nocomb_files = []
    opt_files = []
    optnew_files = []
    #looping on all the dirs and files in the selected directory using os.walk
    for root, dirs, files in os.walk(path_to_neutrino_files):
        # selection of the file type (numuCC, nueCC..)
        if files_key in root:
            for file in files:
                if file.startswith('km3'):
                    #selection of the "standard trigger" files
                    if file.endswith(str(st)):
                        st_files.append(os.path.join(root,file))
                    #selection of the "nocombine" files
                    elif file.endswith(str(nocomb)):
                        nocomb_files.append(os.path.join(root,file))
                    #selection of the "opt" files
                    elif file.endswith(str(opt)):
                        opt_files.append(os.path.join(root,file))
                    elif file.endswith(str(optnew)):
                        optnew_files.append(os.path.join(root,file))
    return st_files, nocomb_files, opt_files, optnew_files






    # Load file function
#The function is used to load the JTE processed files

def load_array_files(files_key, path_to_neutrino_files, st, nocomb, opt, optnew):
   
    """
    Loops on filenames in the path di and saves the filenames in lists according to the selected filters.
    
    Parameters:
    -----------
    files_key : str
        the file type, as 'numuCC', 'numuNC', etc...
    path_to_neutrino_files : str
        the path to the 'neutrino' directory
    st : str
        the extention of the "standard trigger" files
    nocomb : str
        the extention of the "nocombine" files
    opt : str
        the extention of the "optimised" files
    optnew : str
            the extention of the "optimised new" files

    Returns:
    --------
    st_files : list
        list of the paths of each "st" file
    nocomb_files : list
        list of the paths of each "nocomb" file
    opt_files : list
        list of the paths of each "opt" file
    optnew_files : list
    list of the paths of each "optnew" file

    """
    st_out = []
    nocomb_out = []
    opt_out = []
    optnew_out = []
    #looping on all the dirs and files in the selected directory using os.walk
    for root, dirs, files in os.walk(path_to_neutrino_files):
        # selection of the file type (numuCC, nueCC..)
        for file in files:
            if file.startswith(str(files_key)):
                #selection of the "standard trigger" files
                if file.endswith(str(st)):
                    with open(os.path.join(root,file),'rb') as f:
                        st_out = pickle.load(f)
                #selection of the "nocombine" files
                elif file.endswith(str(nocomb)):
                    with open(os.path.join(root,file),'rb') as f:
                        nocomb_out = pickle.load(f)
                #selection of the "opt" files
                elif file.endswith(str(opt)):
                    with open(os.path.join(root,file),'rb') as f:
                        opt_out = pickle.load(f)
                elif file.endswith(str(optnew)):
                    with open(os.path.join(root,file),'rb') as f:
                        optnew_out = pickle.load(f)
    return st_out, nocomb_out, opt_out, optnew_out