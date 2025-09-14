import os
import numpy as np
import datetime
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field

from utils.utils import logger, load_dataloaders, save_dataloaders
from utils.read_data import CHANNEL_NAMES, read_eeg_data, build_dataloader, EEGDataset
from utils.plot_functions import get_confusion_matrix, plot_training_results
from classifier.models import GCN, EEGCN, get_weights
from classifier.training import train, test, train_eegcn, test_eegcn


DATA_FOLDER = '../eeg_data/'            # folder where the dataset is stored
DATASET_FOLDER = './saved_datasets/'    # folder where to save the dataset
RESULTS_FOLDER = './results_classifier/' # folder where to save the results
GRAPH_PATH = '../graph_montage.txt'

@dataclass
class Config:
    """
    A dataclass to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 5              # number of subjects to consider
    first_subj: int = 1                      # first subject to consider
    dataset_size: int = 0                    # size of the dataset
    network_type: str = 'gnn'           
    classification: str = 'ms'               # classification type (cq, ms, both)
    pretrained: bool = False                 # use pretrained model
    nclasses: int = 2                        # number of classes
    nelectrodes: int = len(CHANNEL_NAMES)    # total number of electrodes available
    timewindow: float = 1                  # time window for the spectrogram
    input_channels: int = len(CHANNEL_NAMES) # number of input channels
    channels: list = field(default_factory=lambda: CHANNEL_NAMES) 
    input_data: int = None
    checkpoint_path: str = None              # e.g. './results_classifier/resnet18_20231119-152229'
    optimizer: optim = optim.Adam
    learning_rate: float = 0.001
    loss_fn: nn = nn.CrossEntropyLoss 
    batch_size: int = 32                   
    epochs: int = 3               
    train_rate: float = 0.8
    valid_rate: float = 0.1
    graph_path: str = GRAPH_PATH
    n_cnn: int = 3                           # number of 1D convolutions to extract features from a signal, >=2
    n_mp: int = 1                            # hop distance in graph to collect information from,
    aggregate: str = 'mean'                  # aggregation method for graph convolution layers
    d_hidden: int = 60                       # number of hidden channels of graph convolution layers
    d_latent: int = 90               
    activation: str = 'tanh'                 # activation function to use, [Leaky ReLU, ReLU, Tanh]
    pooling: str = 'max'                     # pooling strategy to use, [Max, Average]
    kernel_size: int = 30                    # kernel size for the 1D convolutions
    norm_enc: int = 1                        # whether to use batch normalization after each 1D convolution (1) or not (0)
    norm_proc: str = 'graph'                 # normalization for the processing layers, [none, batch, graph, layer]
    p_dropout: float = 0.0                   # dropout probability
    normalization: str = 'minmax'            # normalization for the input data
    key: int = 0                             # key to identify the model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')



def run(dataset: EEGDataset):
    """
    Train a neural network classifier on the given dataset
    """
    # make directory to save results
    os.mkdir(RESULTS_FOLDER) if not os.path.isdir(RESULTS_FOLDER) else None
    os.mkdir(CONFIG.dir_path) if not os.path.isdir(CONFIG.dir_path) else None

    # save configuration parameters
    file_path = CONFIG.dir_path + '/config.txt'
    CONFIG.save_config(file_path)

    # build data loaders
    logger.info("Building data loaders...")

    if CONFIG.checkpoint_path is not None:
        # load dataloaders from file
        file_dataloaders = CONFIG.checkpoint_path + '/dataloaders.pkl'
        dataloaders = load_dataloaders(file_dataloaders)
        trainloader, validloader, testloader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    else:
        trainloader, validloader, testloader = build_dataloader(dataset, batch_size=CONFIG.batch_size, graph_path=CONFIG.graph_path, train_rate=CONFIG.train_rate, 
                                                                valid_rate=CONFIG.valid_rate, shuffle=True, resample=False) 
    dataloaders = {'train': trainloader, 'val': validloader, 'test': testloader}
    
    # get the first batch of the trainloader and print some information
    batch = next(iter(trainloader))
    spectrogram = batch.x[0]
    logger.info(f'Spectrogram shape: {spectrogram.shape}')
    logger.info(f'Min spectrogram: {torch.min(spectrogram)}')
    logger.info(f'Max spectrogram: {torch.max(spectrogram)}')

    # save dataloaders to file
    file_dataloaders = CONFIG.dir_path + '/dataloaders.pkl'
    save_dataloaders(dataloaders, file_dataloaders)

    num_node_features = spectrogram.shape[0] 
    # load model
    if CONFIG.classification in ['ms', 'cq']:
        num_classes = 2
    else:
        num_classes = 3
        
    if CONFIG.network_type == 'gnn':
        model = GCN(num_node_features, num_classes=num_classes, hidden_channels=64)
    else:
        model = EEGCN(CONFIG)

    # define loss function and optimizer
    class_weights = get_weights(dataset, CONFIG.nclasses)
    loss_fn = CONFIG.loss_fn(weight=class_weights, reduction='mean')
    logger.info(f'Weights for each class: {class_weights}')
    optimizer = CONFIG.optimizer(model.parameters(), lr=CONFIG.learning_rate)
    
    # train model
    logger.info("Training the model...")
    file_checkpoint = CONFIG.checkpoint_path + '/checkpoint.pt' if CONFIG.checkpoint_path is not None else None
    load = True if CONFIG.checkpoint_path is not None and os.path.isfile(file_checkpoint) else False
    if CONFIG.network_type == 'gnn':
        model = train(model, optimizer, loss_fn, dataloaders, num_epochs=CONFIG.epochs, 
                            folder=CONFIG.dir_path, load_checkpoint=load, 
                            checkpoint_path=file_checkpoint, device=CONFIG.device)
        # test the model
        logger.info("Testing the model...")
        test_acc = test(model, testloader, folder = CONFIG.dir_path)
        logger.info(f'Test accuracy: {test_acc}')
    else:
        model = train_eegcn(model, optimizer, dataloaders, loss_fn, CONFIG.device, CONFIG.epochs, folder=CONFIG.dir_path)

    plot_training_results(f'{CONFIG.dir_path}/results.txt')

    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=2, help='number of subjects for which the correlation is computed')
    parser.add_argument('-nt', '--network_type', type=str, default='eegcn', help='network type (shallownet, resnet18)')
    parser.add_argument('-ct', '--classification', type=str, default='ms', help='classification type (cq, ms, both)')
    parser.add_argument('-ic', '--input_channels', type=int, default=len(CHANNEL_NAMES), help='number of channels in dataitem')
    parser.add_argument('-ch', '--channels', type=lambda s: [str(item).upper() for item in s.split(',')], default=CHANNEL_NAMES, help='channels for which to compute the masks')
    parser.add_argument('-cp', '--checkpoint_path', type=str, default=None, help='path to the checkpoint to load')
    #parser.add_argument('-tw', '--timewindow', type=int, default=1, help='time window for the spectrogram')
    parser.add_argument('--n_cnn', type=int, default=3, help='Number of 1D convolutions to extract features from a signal, >=2')
    parser.add_argument('--n_mp', type=int, default=1, help='Hop distance in graph to collect information from, >=1')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['none', 'eq', 'mean', 'max'],)
    parser.add_argument('--d_hidden', type=int, default=60, help='Number of hidden channels of graph convolution layers')
    parser.add_argument('--d_latent', type=int, default=90, help='Number of features to extract from a EEG signal')
    parser.add_argument('--activation', type=str, default='tanh', choices=['leaky_relu', 'relu', 'tanh'], help='Activation function to use, [Leaky ReLU, ReLU, Tanh]')
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg'], help='Pooling strategy to use, [Max, Average]')
    parser.add_argument('--kernel_size', type=int, default=30)
    parser.add_argument('--norm_enc', type=int, default=1, choices=[0,1],)
    parser.add_argument('--norm_proc', type=str, default='graph', choices=['none', 'batch', 'graph', 'layer'],)
    parser.add_argument('--p_dropout', type=float, default=0., help='Dropout probability')
    parser.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 's', 'z', 'f'],)
    parser.add_argument('--key', type=int, default=0)
    args = parser.parse_args()
    args.network_type = args.network_type.lower()

    # check that configuration parameters are consistent
    assert args.input_channels == len(args.channels) or args.input_channels == 1, "Error: number of input channels must be equal to the number of channels or 1 if each channel is used as a separate input"

    # set the number of classes
    if str(args.classification) == 'both':
        args.nclasses = 3
    else:
        args.nclasses = 2

    # initialize configuration parameters
    CONFIG = Config(**args.__dict__)
    CONFIG.dir_path = f'{RESULTS_FOLDER}/{CONFIG.network_type}_{CONFIG.classification}_{CONFIG.curr_time}' # directory to save classifier results

    dataset = read_eeg_data(DATA_FOLDER, DATASET_FOLDER, input_channels=CONFIG.input_channels, 
                            number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification, 
                            channel_list = CONFIG.channels, time_window = CONFIG.timewindow)
    
    
    # add a certain number of samples (all zeros) to the dataset with label 2 (fictitious class)
    """ for i in range(1000):
        dataset.spectrograms.append(np.zeros(dataset.spectrograms[0].shape))
        dataset.labels.append(2)
        dataset.raw = list(dataset.raw)
        dataset.raw.append(np.zeros(dataset.raw[0].shape))
        dataset.channel.append(list(dataset.channel[0]))
        dataset.id.append(list(dataset.id[0]))
        dataset.edge_index.append(list(dataset.edge_index[0]))
        CONFIG.nclasses = 3 """
        
    CONFIG.dataset_size = len(dataset)
    logger.info(dataset.spectrograms[0].shape)
    logger.info(f'Dataset length: {len(dataset)}')
    logger.info(f'Unique labels: {np.unique(dataset.labels)}')
    

    # train a neural network classifier on the given dataset
    run(dataset)
    
