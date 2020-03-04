import math
import random
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter

from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops

### ================== Definition of custom GCN

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
#         tensor.data.fill_(1.0)   # trivial example
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class custom_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        
        fill_value = 1 if not improved else 2
        
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        
        row, col = edge_index   
        # row includes the starting points of the edges  (first row of edge_index)
        # col includes the ending points of the edges   (second row of edge_index)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # row records the source nodes, which is the index we are trying to add
        # deg will record the out-degree of each node of x_i in all edges (x_i, x_j) including self_loops
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
#         print('whole GCN training normalized_edge_weight: \n', normalized_edge_weight)
        return edge_index, normalized_edge_weight

    def forward(self, x, edge_index, edge_weight = None):
        """"""
#         print('current weight is: ')
#         print(self.weight)
#         print('current bias is: ')
#         print(self.bias)
        
        x = torch.matmul(x, self.weight)   # update x (embeddings)
        
#         print('inside custom_GCN, edge_index: ', edge_index.shape, '\n', edge_index)
        res = self.propagate(edge_index, x = x, norm = edge_weight)
        return res

    # self is the first parameter of the message func
    def message(self, x_j, norm):
        # in source code of the MessagePassing:
#         self.__message_args__ = getargspec(self.message)[0][1:]  : will be initialized as [x_j, norm]
        
        # view is to reshape the tensor, here make it only a single column
        # use the normalized weights multiplied by the feature of the target nodes
        '''
        For each of extended edge_index:(x_i, x_j), assume there is N such edges
        x_j of shape (N, k) , assume there is k features, value along each row are the same
        norm of shape (1, m), assume there is m edges (including self loops), 1-D tensor
        '''
#         print('inside the message custom_GCN: norm \n', norm.shape, '\n', norm)
#         print('inside the message custom_GCN: x_j \n', x_j.shape, '\n', x_j)
        res = norm.view(-1, 1) * x_j  # use the element wise multiplication
        return res

    def update(self, aggr_out):
        # update the embeddings of each node
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



### ====================== Establish a GCN based model ========================
class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
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
    def __init__(self, in_channels, out_channels, input_layers = [16, 16], dropout=0.3):
        """
        input layers: list of integers
        dropout: probability of droping out 
        """
        super(Net, self).__init__()
        # one trivial example
#         self.conv1 = custom_GCNConv(in_channels, out_channels)
#         self.conv2 = GCNConv(16, dataset.num_classes)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_layers = input_layers
        self.dropout = dropout
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        self.input_layers = [self.in_channels] + self.input_layers + [self.out_channels]
        for i, _ in enumerate(self.input_layers[:-1]):
            self.layers.append(custom_GCNConv(self.input_layers[i],self.input_layers[i+1]))
        self.layers = ListModule(*self.layers)

    # change the dropout positions: 
    def forward(self, edge_index, features, edge_weights = None):
        if len(self.layers) > 1:
            for i in range(len(self.layers)-1):
                features = F.relu(self.layers[i](features, edge_index, edge_weights))
#                 if i>0:
                features = F.dropout(features, p = self.dropout, training = self.training)
                    
            features = self.layers[len(self.layers)-1](features, edge_index, edge_weights)
        else:
            features = self.layers[0](features, edge_index, edge_weights)    # for a single layer case

        predictions = F.log_softmax(features, dim=1)
        return predictions

class single_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, input_layers = [], dropout=0.3):
        """
        input layers: list of integers
        dropout: probability of droping out 
        """
        super(single_Net, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        # here we just initialize the model
        self.conv1 = custom_GCNConv(self.in_channels, self.out_channels)
        

    def forward(self, edge_index, features, edge_weights = None):
        # call the instance of the custom_GCNConv
        z = self.conv1(features, edge_index, edge_weights)    # for a single layer case, z is embeddings
#         print('embeddings inside the net work model, result is: \n', z)
        
        predictions = F.log_softmax(z, dim=1)
#         print('calibration inside the net work model, result is: \n', predictions)
        return predictions