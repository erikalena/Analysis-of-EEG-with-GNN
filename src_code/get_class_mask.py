import os
import numpy as np
import datetime
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from classifier.models import EEGCN
from dataclasses import dataclass, field
from utils.utils import logger
from utils.read_data import read_eeg_data, build_dataloader, CHANNEL_NAMES

 
DATA_FOLDER = '../eeg_data/'            # folder where the dataset is stored
with open(os.path.join(DATA_FOLDER, 'subject-info.csv'), 'r') as csv:
    COUNT_QUALITY = [line.strip().split(',')[-1] for line in csv.readlines()][1:]
DATASET_FOLDER = './saved_datasets/'    # folder where to save the dataset
RESULTS_FOLDER = './results_classifier/' # folder where to save the results

@dataclass
class Config:
    """
    A dataclass to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    timewindow: float = 5
    number_of_subjects: int = 5                 # number of subjects to consider
    network_type: str = 'eegcn'              # network type (shallownet, resnet18, eegcn)
    input_channels: int = len(CHANNEL_NAMES)    # number of input channels
    channels: list = field(default_factory=lambda: CHANNEL_NAMES)
    classification: str = 'ms'                  # classification type (cq, ms, both)
    batch_size: int = 1                        # batch size for the training of the classifier with additional mask layer
    nclasses: int = 2                        # number of classes
    n_cnn: int = 3                           # number of 1D convolutions to extract features from a signal, >=2
    n_mp: int = 1                            # hop distance in graph to collect information from,
    aggregate: str = 'mean'                  # aggregation method for graph convolution layers
    d_hidden: int = 64                       # number of hidden channels of graph convolution layers
    d_latent: int = 30
    activation: str = 'tanh'                 # activation function to use, [Leaky ReLU, ReLU, Tanh]
    pooling: str = 'max'                     # pooling strategy to use, [Max, Average]
    kernel_size: int = 15                    # kernel size for the 1D convolutions
    norm_enc: int = 0                        # whether to use batch normalization after each 1D convolution (1) or not (0)
    norm_proc: str = 'batch'                 # normalization for the processing layers, [none, batch, graph, layer]
    p_dropout: float = 0.0                   # dropout probability
    normalization: str = 'minmax'            # normalization for the input data
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


@torch.no_grad()
def guess_target(model, x, edge_index):
    """
    Get the predicted class for the input graph.
    """
    out = model(x, edge_index)
    return out.argmax(dim=-1).item()

def tv_2d(m):
    """
    Simple total-variation on a matrix mask m in [0,1]
    """
    tv_h = (m[:, 1:] - m[:, :-1]).abs().mean()
    tv_v = (m[1:, :] - m[:-1, :]).abs().mean()
    return tv_h + tv_v



def infer_edge_mask(model, dataset, batch_size, num_iters=500, lr=0.01, l1_coeff=0.1, tv_coeff=0.2, device=None, folder=None):
    """
    The objective is to learn a minimal subgraph
    which maintains the same classification.

    input:
        model: the pretrained EEGCN model
        dataset: A list of (x, y) tuples representing the graphs
        batch_size: batch size established for dataloader
        num_iters: number of optimization iterations
        lr: Learning rate
        l1_coeff: Coefficient for sparsity regularization
        tv_coeff: Coefficient for total variation regularization
        device: Device to run on

    output:
        edge_weights: mask weights for graph's edges
    """
    model.eval()
    device = device if device is not None else next(model.parameters()).device
    dataloader, _, _ = build_dataloader(dataset, test_idx=None, batch_size=batch_size, shuffle=False, train_rate=1.0, valid_rate=0.0, folder=folder)

    data = next(iter(dataloader))
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    edge_mask_param = torch.nn.Parameter(torch.randn(num_edges, device=device))
    optimizer = torch.optim.Adam([edge_mask_param], lr=lr)

    for iteration in tqdm(range(num_iters)):
        for data in dataloader:
            x = data.x
            edge_index = data.edge_index
            x = x.to(device)
            edge_index = edge_index.to(device)
            with torch.no_grad():
                logits = model(x, edge_index)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred_class = logits.argmax(dim=-1).item()

            # convert edge mask parameters to [0,1] probabilities
            edge_weights = torch.sigmoid(edge_mask_param)

            # Pass all edges with their corresponding weights to the model
            inputs = (data.x.float(), edge_index, edge_weights, data.batch)
            labels_y = data.y.to(device)
            logits = model(*inputs)

            #labels_y = torch.ones(logits.shape)
            target_loss = F.cross_entropy(logits, labels_y)
            sparsity_loss = l1_coeff * edge_weights.mean()
            
            # for TV regularization, convert edge mask to adjacency matrix
            num_nodes = int(data.x.shape[0]/batch_size)
            adj_mask = torch.zeros(num_nodes, num_nodes, device=device)
            adj_mask[edge_index[0], edge_index[1]] = edge_weights
            # make symmetric for undirected graphs
            adj_mask = (adj_mask + adj_mask.T) / 2
            tv_loss = tv_coeff * tv_2d(adj_mask)
            total_loss = target_loss + sparsity_loss + tv_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if iteration % 5 == 0:
            with torch.no_grad():
                pred_class = logits.argmax(dim=-1).item()
                logger.info(f"Iter {iteration}: Loss={total_loss:.4f}, "
                    f"Target Loss={target_loss:.4f}, Sparsity={sparsity_loss:.4f}, "
                    f"TV={tv_loss:.4f}, Pred={pred_class}, "
                    f"Edges kept={edge_weights.mean().item():.3f}")

    significant_edges = (edge_weights > 0.5).sum().item()
    logger.info(f"Final: {significant_edges}/{num_edges} edges with weight > 0.5")
    logger.info(f"Mean edge weight: {edge_weights.mean().item():.4f}")

    return edge_weights.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=6, help='number of subjects for which the correlation is computed')
    parser.add_argument('-ct', '--classification', type=str, default='ms', help='classification type (cq, ms)')
    parser.add_argument('-ic', '--input_channels', type=int, default=len(CHANNEL_NAMES), help='number of channels in dataitem')
    parser.add_argument('--n_cnn', type=int, default=3, help='Number of 1D convolutions to extract features from a signal, >=2')
    parser.add_argument('--n_mp', type=int, default=1, help='Hop distance in graph to collect information from, >=1')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['none', 'eq', 'mean', 'max'],)
    parser.add_argument('--d_hidden', type=int, default=64, help='Number of hidden channels of graph convolution layers')
    parser.add_argument('--d_latent', type=int, default=100, help='Number of features to extract from a EEG signal')
    parser.add_argument('--activation', type=str, default='relu', choices=['leaky_relu', 'relu', 'tanh'], help='Activation function to use, [Leaky ReLU, ReLU, Tanh]')
    parser.add_argument('--pooling', type=str, default='max', choices=['max', 'avg'], help='Pooling strategy to use, [Max, Average]')
    parser.add_argument('--kernel_size', type=int, default=30)
    parser.add_argument('--norm_enc', type=int, default=1, choices=[0,1],)
    parser.add_argument('--norm_proc', type=str, default='graph', choices=['none', 'batch', 'graph', 'layer'],)
    parser.add_argument('--p_dropout', type=float, default=0.2, help='Dropout probability')
    args = parser.parse_args()
    args.nclasses = 2

    # initialize configuration parameters
    CONFIG = Config(**args.__dict__)

    CONFIG.dir_path = f'{RESULTS_FOLDER}/{CONFIG.network_type}_{CONFIG.classification}_{CONFIG.curr_time}' # directory to save classifier results


    # get instances of class 1
    dataset = read_eeg_data(DATA_FOLDER, DATASET_FOLDER, input_channels=CONFIG.input_channels,
                            number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification,
                            channel_list = CONFIG.channels, time_window = CONFIG.timewindow)
    
    # split dataset with respect to the two classes
    dataset_class_0 = dataset.select_class(0)
    dataset_class_1 = dataset.select_class(1)
    logger.info(f'Size of class 1 dataset: {len(dataset_class_1)}')
    logger.info(f'Size of class 0 dataset: {len(dataset_class_0)}')

    # load model from training folder
    training_folder = './results_classifier/eegcn_ms_20250922-074324'
    model = EEGCN(CONFIG)
    model.load_state_dict(torch.load(f'{training_folder}/model.pt'))
    model.eval()

    # load graph edges from txt file
    edge_index = np.loadtxt('../graph_montage.txt', delimiter=',')
    edge_index = torch.tensor(edge_index).long()

    # learn masks for class 0 and class 1
    mask0 = infer_edge_mask(model, dataset_class_1, batch_size=CONFIG.batch_size, num_iters=30, folder=DATASET_FOLDER)
    mask1 = infer_edge_mask(model, dataset_class_0, batch_size=CONFIG.batch_size, num_iters=30, folder=DATASET_FOLDER)

    # save masks to file
    np.savetxt(f'{CONFIG.classification}_mask_class_0.txt', mask0.numpy(), delimiter=',')
    np.savetxt(f'{CONFIG.classification}_mask_class_1.txt', mask1.numpy(), delimiter=',')