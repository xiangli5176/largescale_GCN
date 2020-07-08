import math
import random
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter

from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops

from clustergcn_conv_enhance import GCNConv


############################################
########## Customized Neuro Nets ###########
############################################




### ====================== Establish a GCN based model ========================
class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
            Module initializing.
            Add positional arguments
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, input_layers = [16, 16], dropout = 0.3, improved = False, diag_lambda = -1):
        """
        input layers: list of integers
        dropout: probability of droping out 
        """
        super(Net, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_layers = input_layers
        self.dropout = dropout
        self.improved = improved 
        self.diag_lambda = diag_lambda
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        # store the neuron number on each layer
        self.input_layers = [self.in_channels] + self.input_layers + [self.out_channels]
        
        # first layer is to convert the graph into the first embedding:
#         self.layers.append(GCNConv(2 * self.input_layers[0], self.input_layers[1]) )
        for i in range(len(self.input_layers)-1):
            self.layers.append(GCNConv(2 * self.input_layers[i], self.input_layers[i+1], 
                                       improved = self.improved, diag_lambda = self.diag_lambda, dropout = self.dropout))
        self.layers = ListModule(*self.layers)
        
        self.bn_layers = []
        for i in range(1, len(self.input_layers)):
            self.bn_layers.append(torch.nn.BatchNorm1d(self.input_layers[i]))
        self.bn_layers = ListModule(*self.bn_layers)
    
    
    
    # change the dropout positions: 
    def forward(self, edge_index, features):
        if len(self.layers) > 1:
            for i in range(len(self.layers)-1):
                features = self.layers[i](features, edge_index, dropout_training=self.training)
#                 features = F.dropout(features, p = self.dropout, training = self.training)
                features = self.bn_layers[i](features)
                features = F.relu(features)
                    
            features = self.layers[len(self.layers)-1](features, edge_index, dropout_training=self.training)
        else:
            features = self.layers[0](features, edge_index, dropout_training=self.training)    # for a single layer case
        
        predictions = features
        # just use the linear layer output, since we are using the cross-entropy loss
        # just pay attention to the test part, change the predictions
        
#         predictions = F.log_softmax(features, dim=1)
        # if using the nll loss, then we need this log_softmax layer
        
        return predictions