import torch
from utils.read_data import EEGDataset
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool



class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def get_weights(dataset: EEGDataset, nclasses: int):
    """
    Function to get weights for each class
    proportional to the inverse of the number of samples
    """
    counts = [sum([1 for i, _ in enumerate(dataset) if dataset.get_label(i) == label]) for label in range(nclasses)]

    # compute class weights
    class_weights = [sum(counts)/(nclasses **2 * counts[i]) for i in range(nclasses)]

    return torch.tensor(class_weights).float()



