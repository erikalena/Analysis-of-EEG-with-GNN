import os
from matplotlib import pyplot as plt
import pickle
import mne
import numpy as np
import copy

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


##################################################
# Utility functions for training and saving models
##################################################


def load_dataset(data_path: str):
    """
    Load dataset from file
    """
    dataset = None
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            logger.info(data_path)
            dataset = pickle.load(f)
        return dataset
    else:
        logger.error("Error: file not found")
        exit(0)

def save_dataloaders(dataloaders: dict, file_path: str):
    """
    Save dataloaders to file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)

def load_dataloaders(file_path: str):
    """
    Load dataloaders from file
    """
    with open(file_path, 'rb') as f:
        dataloaders = pickle.load(f)
    return dataloaders


#######################################################################
## Utility functions to work with EEG data and plot topographic maps ##
#######################################################################


def get_montage(raw):

    # montage available in mne
    montage = mne.channels.make_standard_montage('standard_1020')

    # make a copy of montage positions
    montage_pos = copy.deepcopy(montage._get_ch_pos())

    montage_pos = {k.upper(): v for k, v in montage_pos.items()}
    electrode_dicts = {ch_name: montage_pos[ch_name] for ch_name in raw.info['ch_names']}

    # get fiducial points
    fid = montage.dig
    nasion = fid[1]['r']  # Nasion point
    lpa = fid[0]['r']  # Left point
    rpa = fid[2]['r']  # Right point

    custom_montage = mne.channels.make_dig_montage(nasion = nasion, lpa=lpa, rpa=rpa, ch_pos = electrode_dicts)

    return custom_montage

    
def get_topographic_map(filename: str, size: int=500) -> tuple:
    """
    Function to obtain the positions (montage) of the electrodes with respect
    to the given edf file.
    If plot is True, it plots the topographic map of the gamma band.
    - filename: a random edf file from the dataset
    - size: number of data points in the time interval to plot
    """
    # read file from folder
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False);
    
    # standardize channel names
    raw.rename_channels(lambda x: x.upper()[4:])

    # remove ECG channel
    raw.drop_channels(['ECG'])
    # and name A2-A1 as A2
    #raw.rename_channels({'A2-A1': 'A2'})
    raw.drop_channels(['A2-A1'])

    # set montage
    custom_montage = get_montage(raw)
    raw.set_montage(custom_montage)
    
    # get 2d positions of electrodes
    pos_2d = mne.viz.topomap._get_pos_outlines(raw.info, picks=None, sphere=None)[0]
    positions = np.zeros((len(raw.ch_names), 2))
    for ch in raw.ch_names:

        idx = raw.ch_names.index(ch)
        positions[raw.ch_names.index(ch), :] = [pos_2d[idx][0], pos_2d[idx][1]-0.02]

    # extract data to plot
    data = np.zeros((len(raw.ch_names), size))

    
    for i, ch in enumerate(raw.ch_names):
        data[i, :] = raw[ch][0][0][:size]

    return raw, positions

