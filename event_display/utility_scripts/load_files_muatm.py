import ROOT
from ROOT import gSystem
import root_numpy as rnp
import numpy as np
import os
import pickle


# Load file function
#The function is used to load the JTE processed files

def load_root_files(files_key, path_to_neutrino_files, ext1, ext2, ext3, ext4):
    """
    Loops on filenames in the path di and saves the filenames in lists according to the selected filters.
    
    Parameters:
    -----------
    files_key : str
        the file type, as 'numuCC', 'numuNC', etc...
    path_to_neutrino_files : str
        the path to the 'neutrino' directory
    ext1 : str
        the extention of the first group of files
    ext2 : str
        the extention of the second group of files
    ext3 : str
        the extention of the third group of files
    ext4 : str
        the extention of the forth group of files

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
    ext1_files = []
    ext2_files = []
    ext3_files = []
    ext4_files = []
    #looping on all the dirs and files in the selected directory using os.walk
    for root, dirs, files in os.walk(path_to_neutrino_files):
        # selection of the file type (numuCC, nueCC..)
        for file in files:
            if file.startswith(str(files_key)):
                #selection of the "ext1" files
                if file.endswith(str(ext1)):
                    ext1_files.append(os.path.join(root,file))
                #selection of the "ext2" files
                elif file.endswith(str(ext2)):
                    ext2_files.append(os.path.join(root,file))
                #selection of the "ext3" files
                elif file.endswith(str(ext3)):
                    ext3_files.append(os.path.join(root,file))
                #selection of the "ext4" files
                elif file.endswith(str(ext4)):
                    ext4_files.append(os.path.join(root,file))
    return ext1_files, ext2_files, ext3_files, ext4_files






    # Load file function
#The function is used to load the JTE processed files

def load_array_files(files_key, path_to_neutrino_files, ext1, ext2, ext3, ext4):
    """
    Loops on filenames in the path di and saves the filenames in lists according to the selected filters.
    
    Parameters:
    -----------
    files_key : str
        the file type, as 'numuCC', 'numuNC', etc...
    path_to_neutrino_files : str
        the path to the 'neutrino' directory
    ext1 : str
        the extention of the first group of files
    ext2 : str
        the extention of the second group of files
    ext3 : str
        the extention of the third group of files
    ext4 : str
        the extention of the forth group of files

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

    ext1_out = []
    ext2_out = []
    ext3_out = []
    ext4_out = []

    #looping on all the dirs and files in the selected directory using os.walk
    for root, dirs, files in os.walk(path_to_neutrino_files):
        # selection of the file type (numuCC, nueCC..)
        for file in files:
            if file.startswith(str(files_key)):
                #selection of the "ext1" files
                if file.endswith(str(ext1)):
                    with open(os.path.join(root,file),'rb') as f:
                        ext1_out = pickle.load(f)
                #selection of the "ext2" files
                elif file.endswith(str(ext2)):
                    with open(os.path.join(root,file),'rb') as f:
                        ext2_out = pickle.load(f)
                #selection of the "ext3" files
                elif file.endswith(str(ext3)):
                    with open(os.path.join(root,file),'rb') as f:
                        ext3_out = pickle.load(f)
                elif file.endswith(str(ext4)):
                    with open(os.path.join(root,file),'rb') as f:
                        ext4_out = pickle.load(f)
    return ext1_out, ext2_out, ext3_out, ext4_out