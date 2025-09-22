import numpy as np
import datetime
import argparse
import torch
import torch.nn.functional as F
from classifier.models import EEGCN
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import logger

@dataclass
class Config:
    """
    A dataclass to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    #if isinstance(out, (tuple, list)):  # some models return (logits, att_adj)
    #    out = out[0]
    return out.argmax(dim=-1).item()

def tv_2d(m):
    """
    Simple total-variation on a matrix mask m in [0,1]
    """
    tv_h = (m[:, 1:] - m[:, :-1]).abs().mean()
    tv_v = (m[1:, :] - m[:-1, :]).abs().mean()
    return tv_h + tv_v

def edge_index_to_adj(edge_index, num_nodes, device=None):
    """
    Convert edge_index to adjacency matrix
    """
    if device is None:
        device = edge_index.device
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj

def adj_to_edge_index(adj, threshold=1e-6):
    """
    Convert adjacency matrix to edge_index, filtering by threshold
    """
    rows, cols = torch.where(adj > threshold)
    return torch.stack([rows, cols], dim=0)

def infer_edge_mask(model, x, edge_index, target_class=None, num_iters=500, lr=0.1, l1_coeff=0.01, tv_coeff=0.2, device=None):
    """
    The objective is to learn a minimal subgraph 
    which maintains the same classification.
    
    input:
        model: The graph neural network model
        x: Node features (N, C) or (1, C, T)
        edge_index: Edge indices (2, E)
        target_class: Target class to maintain (if None, use model's prediction)
        num_iters: Number of optimization iterations
        lr: Learning rate
        l1_coeff: Coefficient for sparsity regularization
        tv_coeff: Coefficient for total variation regularization
        device: Device to run on
        
    output:
        final_edge_mask (E,): Continuous edge importance scores in [0,1] for each original edge
        masked_edge_index (2, E'): Binary subgraph containing only edges with mask > 0.5
    """
    model.eval()
    
    if device is None:
        device = edge_index.device
    
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    # verify that the instance is correctly classified
    with torch.no_grad():
        logits = model(x, edge_index)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred_class = logits.argmax(dim=-1).item()
        logger.info(f"Initial prediction: {pred_class}")
        if target_class is not None and target_class != pred_class:
            logger.info(f"Warning: provided target_class {target_class} does not match model prediction {pred_class}. Using predicted class.")
            target_class = pred_class
            return None, None
    
    num_edges = edge_index.size(1)
    num_nodes = x.size(-2) if x.dim() == 3 else x.size(1)
    logger.info(num_nodes, num_edges)
    # Initialize edge mask parameters
    edge_mask_param = torch.nn.Parameter(torch.randn(num_edges, device=device) * 0.1)
    optimizer = torch.optim.Adam([edge_mask_param], lr=lr)
    
    # Get target class if not provided
    if target_class is None:
        target_class = guess_target(model, x, edge_index)
        logger.info(f"Target class: {target_class}")
    
    for iteration in tqdm(range(num_iters)):
        # Convert edge mask parameters to [0,1] probabilities
        edge_mask = torch.sigmoid(edge_mask_param)
        
        # Use Gumbel sampling for differentiable edge selection
        edge_probs = edge_mask.unsqueeze(0)  # (1, E)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(edge_probs) + 1e-20) + 1e-20)
        edge_logits = torch.log(edge_probs + 1e-20) + gumbel_noise
        hard_mask = torch.sigmoid(edge_logits / 0.1)  # temperature=0.1 for sharp selection
        
        # Apply hard mask to create subgraph
        selected_edges = (hard_mask.squeeze() > 0.5)
        if selected_edges.sum() > 0:
            sub_edge_index = edge_index[:, selected_edges]
            logits = model(x, sub_edge_index)
        else:
            # If no edges selected, return zero logits
            logits = torch.zeros(1, model.num_classes if hasattr(model, 'num_classes') else 10, device=device)
    
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        
        # Compute loss
        target_loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))
        sparsity_loss = l1_coeff * edge_mask.mean()  # Encourage fewer edges
        
        # For TV regularization, convert edge mask to adjacency matrix
        adj_mask = torch.zeros(num_nodes, num_nodes, device=device)
        adj_mask[edge_index[0], edge_index[1]] = edge_mask
        # Make symmetric for undirected graphs
        adj_mask = (adj_mask + adj_mask.T) / 2
        tv_loss = tv_coeff * tv_2d(adj_mask)
        
        total_loss = target_loss + sparsity_loss + tv_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            with torch.no_grad():
                pred_class = logits.argmax(dim=-1).item()
                logger.info(f"Iter {iteration}: Loss={total_loss:.4f}, "
                      f"Target Loss={target_loss:.4f}, Sparsity={sparsity_loss:.4f}, "
                      f"TV={tv_loss:.4f}, Pred={pred_class}, "
                      f"Edges kept={edge_mask.mean().item():.3f}")
    
    # Final results
    with torch.no_grad():
        final_edge_mask = torch.sigmoid(edge_mask_param)
        
        # Create thresholded edge index (keep edges with mask > 0.5)
        keep_edges = final_edge_mask > 0.5
        masked_edge_index = edge_index[:, keep_edges] if keep_edges.sum() > 0 else torch.empty((2, 0), device=device, dtype=edge_index.dtype)
        
        logger.info(f"Final: {keep_edges.sum().item()}/{num_edges} edges kept")
        
    return final_edge_mask.detach(), masked_edge_index



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
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
        
    # load model from training folder
    training_folder = './eegcn_ms_20250922-074324'
    model = EEGCN(CONFIG)
    model.load_state_dict(torch.load(f'{training_folder}/model.pt'))
    model.eval()

    # load dataloaders from training folder pkl file
    import pickle
    with open(f'{training_folder}/dataloaders.pkl', 'rb') as f:
        dataloaders = pickle.load(f)

    masks = {'class_0': [], 'class_1': []}  # assuming binary classification
    edge_index = np.loadtxt('../graph_montage.txt', delimiter=',')
    edge_index = torch.tensor(edge_index).long()
 
    for data in dataloaders['train']:
        x, y = data.x, data.y
        batch_size = data.num_graphs
        N = x.shape[0] // batch_size
        for i in range(batch_size):
            xi = x[i*N:(i+1)*N, :]
            yi = y[i]
            final_edge_mask, masked_edge_index = infer_edge_mask(model, xi, edge_index, num_iters=50, target_class=yi)[0].cpu()
            masks[f'class_{yi.item()}'].append(masked_edge_index)
       
    # Now masks['class_0'] and masks['class_1'] contain the learned masks for each class
    # You can average them or visualize them as needed
    # Example: average mask per class
    avg_mask_class_0 = torch.stack(masks['class_0']).mean(dim=0)
    avg_mask_class_1 = torch.stack(masks['class_1']).mean(dim=0)

    # plot average masks
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Average Mask Class 0')
    plt.imshow(avg_mask_class_0, cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Average Mask Class 1')
    plt.imshow(avg_mask_class_1, cmap='viridis')
    plt.colorbar()
    plt.show()
    plt.savefig(f'{training_folder}/average_masks.png')