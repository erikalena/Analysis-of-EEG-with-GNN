import os
import numpy as np
import pandas as pd
import mne
import copy
import pickle
import scipy
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from utils.utils import logger

CHANNEL_NAMES = ['FP1','FP2', 'F3','F4','F7','F8','T3','T4','C3','C4',
            'T5','T6','P3','P4','O1','O2','FZ','CZ','PZ','A2']
RECORDING_LENGTH = 60 # seconds (use only the first minute of each recording)


class EEGDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the EEG data
    Input:
        spectrograms: a list of spectrograms
        raw: a list of raw eeg data
        labels: a list of labels
        id: a list of subject ID and recording ID
        channel: a list of channel names
        info: a dictionary containing all the information about the dataset
    """

    def __init__(self, spectrograms, raw, labels, id, channel, edge_index, info):
        self.spectrograms = spectrograms # spectrograms
        self.raw = raw # raw eeg data
        self.labels = labels 
        self.id = id  # subject ID and recording ID
        self.channel = channel # channel name
        self.edge_index = edge_index
        self.info = info
        

    def __len__(self):
        return len(self.spectrograms)
    
    def __getinfo__(self):
        # get info attached to eeg dataset, which are necessary 
        # to recover a bunch of information about the dataset itself
        return self.info
    
    def __getshape__(self):
        # get shape of the dataset
        return self.spectrograms[0].shape[2]

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.raw[idx], self.labels[idx], self.id[idx], self.channel[idx], self.edge_index[idx]
    
    def get_spectrogram(self, idx):
        return self.spectrograms[idx]
    
    def get_raw(self, idx):
        return self.raw[idx]
    
    def get_label(self, idx):
        return self.labels[idx]
    
    def get_id(self, idx):
        return self.id[idx]
    
    def get_channel(self, idx):
        return self.channel[idx]
    
    def get_edge_index(self, idx):
        return self.edge_index[idx]
    
    def set_spectrogram(self, idx, spectrogram):  
        self.spectrograms[idx] = spectrogram
    
    def set_raw(self, idx, raw):
        self.raw[idx] = raw

    def select_class(self, idx):
        """
        Select only the data of the specified class
        """
        # get indices of the channels to be selected
        indices = [i for i, label in enumerate(self.labels) if label == idx]
        # select only the desired channels
        spectrograms = [self.spectrograms[i] for i in indices] if len(self.spectrograms) > 0 else []
        raw = [self.raw[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        id = [self.id[i] for i in indices]
        channel = [self.channel[i] for i in indices]
        edge_index = [self.edge_index[i] for i in indices]
        
        return EEGDataset(spectrograms, raw, labels, id, channel, edge_index, self.info)
    

    def select_channels(self, ch):
        """
        A function which returns an EEGDataset object with only the selected channels
        """
        # get indices of the channels to be selected
        indices = [i for i, channel in enumerate(self.channel) if channel == ch]
        # select only the desired channels
        spectrograms = [self.spectrograms[i] for i in indices] if len(self.spectrograms) > 0 else []
        raw = [self.raw[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        id = [self.id[i] for i in indices]
        channel = [self.channel[i] for i in indices]
        edge_index = [self.edge_index[i] for i in indices]
        return EEGDataset(spectrograms, raw, labels, id, channel, edge_index, self.info)
    
    def filter_data(self, lowcut, highcut, order=2):
        """
        Filter the EEG data in a specific frequency range
        """
        raw = []
        # filter raw data
        for idx, _ in enumerate(self.raw):
            raw.append(filter_eeg_data(self.raw[idx], fs=500, lowcut=lowcut, highcut=highcut, order=order))

        return EEGDataset(self.spectrograms, raw, self.labels, self.id, self.channel, self.edge_index, self.info)
        
    def remove_item(self, idx):
        del self.spectrograms[idx]
        del self.raw[idx]
        del self.labels[idx]
        del self.id[idx]
        del self.channel[idx]
        del self.info[idx]
        del self.edge_index[idx]


def filter_eeg_data(data: np.array, fs: float, lowcut: float, highcut: float, order: int) -> np.array:
    """
    Filter the EEG data in a specific frequency range
    Input:
        data: raw EEG data
        fs: sampling frequency
        lowcut: lower frequency limit
        highcut: higher frequency limit
        order: order of the filter
    Output:
        filtered_eeg_data: filtered raw EEG data
    """
  
    # Filter data in a specific frequency range 
    # Calculate the Nyquist frequency
    nyquist = 0.5 * fs

    # Design the band-pass filter using the Butterworth filter design
    b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

    # Apply the filter to the EEG data
    filtered_eeg_data = scipy.signal.lfilter(b, a, data)

    return filtered_eeg_data

def get_subject_counting_quality(folder: str) -> np.array:

    # find the csv file in the data folder
    files = os.listdir(folder)
    csv_files = [f for f in files if f.endswith('.csv')]
    assert len(csv_files) == 1, 'Error: no csv file found in the folder or multiple csv files found'

    # read subject count quality from csv file
    df = pd.read_csv(f'{folder}/{csv_files[0]}')

    # read last column containing the quality of counting
    quality = df.iloc[:,-1].values
    
    return quality

def read_eeg_data(folder: str, data_path: str, input_channels: int, number_of_subjects: int = 10, 
                  type: str = 'ms', time_window: float = 1, save_spec: bool = True, channel_list: list = None) -> EEGDataset:
    """
    Create an EEGDataset object using the data in the folder
    Input:
        folder: path to the folder containing the data (the folder must contain a csv file with information about the data)
        data_path: path to the file where to save the dataset
        input_channels: number of input channels
        number_of_subjects: number of subjects to be loaded 
        type: determine which data to be used depending on the task we want to classify
         - 'ms': use data to classify mental state (rest, count)
         - 'cq': use data to classify counting quality (good, bad)
         - 'both': use data to classify both mental state and counting quality (3 classes)
        time_window: length of the time window in seconds
        save_spec: if True, save spectrograms, so that if they are not needed, the size of the dataset is smaller
        channel: if not None, select only the specified channel
    Output: 
        dataset: an EEGDataset object
    """
    # final destination of the dataset
    number_of_classes = 2 if type != 'both' else 3
    file_path = f'{data_path}/eeg_dataset_ns_{number_of_subjects}_ch_{input_channels}_nc_{number_of_classes}_{type}_{str(time_window).replace(".", "")}sec.pkl'

    # mkdir if the folder does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # if the file containing the dataset already exists, load dataset from there
    if os.path.exists(file_path):
        logger.info("Loading dataset...")
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    else:
        logger.info("Creating dataset... this may take a while")
    
    # for each edf file in folder, we save the following information
    ids = [] # subject ID + recording ID + segment ID (e.g. 'Subject03_2_90')
    channels = [] # channel names
    raw = [] # raw eeg data
    spectrograms = [] # corresponding spectrograms
    labels = [] # 0 = rest, 1 = count_g, 2 = count_b (if type = 'both'), 0 = rest, 1 = count (if type = 'ms'), 0 = count_g, 1 = count_b (if type = 'cq')
    
    subject_count_quality = get_subject_counting_quality(folder)

    nchannels = len(CHANNEL_NAMES) 
    
    for subj in range(number_of_subjects):
        if subj > len(subject_count_quality):
            break
        filename1 =  'Subject' + "{0:0=2d}".format(subj) + '_1.edf'
        filename2 =  'Subject' + "{0:0=2d}".format(subj) + '_2.edf'

        files = [filename2] if type == 'cq' else [filename1, filename2]

        for file in files:
            logger.info(f'Processing file {file}')

            id = file.split('.')[0]
            label = int(id.split('_')[1]) - 1
            subject = int(id[7:].split('_')[0])

            if type == 'cq' and label == 1 and subject_count_quality[subject] == 1:
                label = 0
            elif type == 'both' and label == 1 and subject_count_quality[subject] == 1:
                label = 2

            # read edf file
            data = mne.io.read_raw_edf(folder + file, preload=True, verbose=False)
            # frequency of the data
            fs = data.info['sfreq']

            factor = 1/time_window
            segment_length = int(fs/factor) 
            n_segments = int(factor*RECORDING_LENGTH) # number of segments in the recording

            for j in range(int(n_segments/2)):
                j = j*2
                img_eeg = []
                raw_eeg = []
                identifiers = []

                for i in range(len(data.ch_names)-1):
                    if channel_list is not None and CHANNEL_NAMES[i] not in channel_list:
                        continue
                    sample = data.get_data(i)[0]
                    # normalize sample
                    sample = (sample - sample.min()) / (sample.max() - sample.min())
                    eeg_data = sample[j*segment_length:(j+1)*segment_length] 
                    raw_eeg.append(eeg_data)
                    #fft = np.fft.fft(eeg_data)[:100]
                    #raw_eeg.append(np.abs(fft))

                    if save_spec:
                        freqs = np.arange(0.5, 60, 2)
                        ncycles = np.linspace(.5,5,len(freqs))
                        mwt = compute_wavelet_transform(eeg_data, fs=fs, n_cycles=ncycles, freqs=freqs)
                        mwt = scipy.signal.resample(mwt, num=200, axis=1) # resample spectrogram to 200 points so that they occupy less space
                        img_eeg.append(np.abs(mwt))
                    else:
                        img_eeg.append([])

                    identifiers.append(f'{id}_{j}')
                
                # input channels are needed only to choose how data is saved
                # if input_channels = 1, each spectrogram is a data item
                # else a single data item is made of the spectrograms of all channels which will be concatenated 
                if input_channels == 1:
                    spectrograms.extend(np.array(img_eeg))
                    raw.extend(raw_eeg)

                    labels.extend([label]*nchannels) if channel_list is None else labels.extend([label]*len(channel_list))
                    ids.extend(identifiers)
                    channels.extend(CHANNEL_NAMES) if channel_list is None else channels.extend(channel_list)

                else:
                    spectrograms.append(np.array(img_eeg))
                    raw.append(raw_eeg)
        
                    labels.append(label) # get label from event
                    ids.append(identifiers)
                    channels.append(CHANNEL_NAMES) if channel_list is None else channels.append(channel_list)
    
    # load edge index to build the graph
    edge_index = np.loadtxt('../graph_montage.txt', delimiter=',')
    edge_index = torch.tensor(edge_index).long()
    edge_index = [edge_index for _ in range(len(spectrograms))]
    
    dataset = EEGDataset(spectrograms, raw, labels, ids, channels, edge_index, {})

    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    return dataset
    
def group_split_indices(dataset_tmp, train_ids, valid_ids, test_ids, folder):
    """
    Return indices for train, validation, and test sets based on user IDs
    """
    train_idx = [i for i, id in enumerate(dataset_tmp.id) if id[0].split('_')[0] in train_ids]
    valid_idx = [i for i, id in enumerate(dataset_tmp.id) if id[0].split('_')[0] in valid_ids]
    test_idx = [i for i, id in enumerate(dataset_tmp.id) if id[0].split('_')[0] in test_ids]
    with open(os.path.join(folder, 'config.txt'), 'a') as f:
        f.write(f'TRAIN: {train_ids}\n')
        f.write(f'VAL: {valid_ids}\n')
        f.write(f'TEST: {test_ids}\n')
    return train_idx, valid_idx, test_idx
    

def build_dataloader(dataset: EEGDataset, test_idx: int, batch_size: int, train_rate: float = 0.8, valid_rate: float = 0.1, shuffle: bool = True, normalization: str = 'minmax', folder: str = None) -> tuple:
    """
    A function which provides all the dataloaders needed for training, validation and testing
    Input:
        dataset: a custom dataset
        batch_size: the batch size
        train_rate: the percentage of the dataset used for training
        valid_rate: the percentage of the dataset used for validation
        test_rate: the percentage of the dataset used for testing
        shuffle: whether to shuffle the dataset before splitting it
    Output:
        trainloader: a dataloader for training
        validloader: a dataloader for validation
        testloader: a dataloader for testing
    """
   
    dataset_tmp = copy.deepcopy(dataset)
    
    # transform data to tensors if not already
    spectrograms = [None]*len(dataset_tmp)
    raw = [None]*len(dataset_tmp)
    labels = [None]*len(dataset_tmp)
    logger.info(f"Raw has dimension: {np.array(dataset_tmp[0][1]).shape}")
    #min_raw = min(torch.as_tensor(dataset_tmp[i][1]).min().item() for i in range(len(dataset_tmp)))
    #max_raw = max(torch.as_tensor(dataset_tmp[i][1]).max().item() for i in range(len(dataset_tmp)))

        
    for idx in range(len(dataset_tmp.raw)):
        spectrograms[idx] = torch.tensor(dataset_tmp.spectrograms[idx].real).float() if len(dataset_tmp.spectrograms) > 0  else dataset_tmp.spectrograms[idx]
        raw[idx] = torch.tensor(np.array(dataset_tmp.raw[idx])).float() 
        #raw[idx] = (raw[idx] - min_raw)/(max_raw - min_raw)
        labels[idx] = torch.tensor(dataset_tmp.labels[idx]).long() 
    
    dataset_tmp.spectrograms = spectrograms
    dataset_tmp.raw = raw
    dataset_tmp.labels = labels
    
    test_rate = round(1 - train_rate - valid_rate, 2)
    user_ids = [id[0].split('_')[0] for id in dataset_tmp.id]
    
    unique_ids = list(set(user_ids))  # Get unique IDs
    #train_ids, temp_ids = train_test_split(unique_ids, test_size=round(1-train_rate,2), random_state=42)  # 60% train, 40% temp
    #valid_ids, test_ids = train_test_split(temp_ids, test_size=test_rate, random_state=42)
    
    #train_ids are all but test_id
    test_ids = [f'Subject{test_idx:02d}']
    train_ids = [id for id in unique_ids if id not in test_ids]
    valid_ids = []
    valid_rate = 0.0
    train_idx, valid_idx, test_idx = group_split_indices(dataset_tmp, train_ids, valid_ids, test_ids, folder)

    # Prepare data for each split
    train_data_list = prepare_graph_data(train_idx, dataset, normalization)

    valid_data_list = []
    if valid_rate > 0:
        valid_data_list = prepare_graph_data(valid_idx, dataset, normalization)

    test_data_list = []
    if test_rate > 0:
        test_data_list = prepare_graph_data(test_idx, dataset, normalization)

    # Create PyTorch Geometric DataLoaders
    trainloader = DataLoader(train_data_list, batch_size=batch_size, shuffle=shuffle)
    validloader = DataLoader(valid_data_list, batch_size=batch_size, shuffle=shuffle)  if len(valid_data_list) > 0 else None
    testloader = DataLoader(test_data_list, batch_size=batch_size, shuffle=shuffle)  if len(test_data_list) > 0 else None

    # Test the dataloader
    logger.info(f"Individual data object x shape: {train_data_list[0].x.shape if train_data_list[0].x is not None else 'None'}")
    logger.info(f"Individual data object raw type: {type(train_data_list[0].raw)}")

    # Get a batch and check
    batch = next(iter(trainloader))
    logger.info(f"Batch x shape: {batch.x.shape if hasattr(batch, 'x') and batch.x is not None else 'None'}")
    logger.info(f"Batch raw type: {type(batch.raw)}")
    logger.info(f"Batch has x attribute: {hasattr(batch, 'x')}")

    # save plot of an example of raw data
    plt.figure()
    plt.plot(batch.x[0])
    plt.title(f"Raw EEG data - Label: {batch.y[0]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.savefig("raw_eeg_example.png")
    plt.close()
            
    return trainloader, validloader, testloader


def prepare_graph_data(indices, dataset, normalization):
    """
    Prepare data as PyTorch Geometric Data objects
    """
    data_list = [] # vector for data objects
    
    # create Data objects
    for i, idx in enumerate(indices):
        # Get and process spectrogram
        if hasattr(dataset, 'get_spectrogram'):
            spec = dataset.get_spectrogram(idx)
        else:
            spec = dataset.spectrograms[idx] if len(dataset.spectrograms) > 0 else None
    
        if spec is not None:
            if hasattr(spec, 'real'):
                x = torch.tensor(spec.real).float()
            else:
                x = torch.tensor(spec).float()
        else:
            logger.error(f"Spectrogram not found: {idx}")
           
        spec = torch.tensor(dataset.spectrograms[idx]).float()  
        raw = torch.tensor(np.array(dataset.raw[idx])).float()
        y = dataset.labels[idx]
        edge_index = dataset.edge_index[idx]
        id = dataset.id[idx]
       
        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
        """
        if normalization == 'minmax':
            ""
            # create tensor with same shape as raw
            x = torch.tensor(dataset.raw[idx]).float()
            for i, ch in enumerate(raw):
                ch = (ch - ch.min()) / (ch.max() - ch.min())
                x[i] = ch
            ""
            x = (raw - raw.min())/(raw.max() - raw.min())
        elif normalization == 'z':
            x = ((raw - raw.mean())/raw.std())
        elif normalization == 's':
            x = ((raw - raw.mean(-1).unsqueeze(1))/raw.std(-1).unsqueeze(1))
        elif normalization == 'f':
            x = F.normalize(raw)
        """
        # Create PyTorch Geometric Data object
        data_obj = Data(
            x=raw, 
            raw=raw,  # Custom attribute for raw data
            y=y,
            edge_index=edge_index,
            id=id
        )
        # Add any additional attributes if needed
        try:
            data_obj.attr1 = dataset[idx][4]  # Store additional data
            data_obj.attr2 = dataset[idx][5]
        except:
            pass
            
        data_list.append(data_obj)
    
    return data_list
