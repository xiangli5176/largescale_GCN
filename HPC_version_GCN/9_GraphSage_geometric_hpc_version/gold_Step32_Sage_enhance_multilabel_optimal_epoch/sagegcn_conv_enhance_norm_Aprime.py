import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math
import torch.nn.functional as F

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
#         tensor.data.fill_(1.0)   # trivial example
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)



class SAGEConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, diag_lambda = -1, dropout = 0, **kwargs):
        """
            cached : turn on to cache the previous results to avoid repeated calculation
        """
        super(SAGEConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.diag_lambda = diag_lambda
        self.dropout = dropout
        # concatenate the previous embedding, therefore have duplicated in_channels
        self.weight = Parameter(torch.Tensor(2 * in_channels, out_channels))

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
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None, diag_lambda = -1):
        """
            Normalization by  
            A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A').
        """
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        A_prime = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        # print('inside norm, shape of the edge_index and A_prime are:')
        # print(type(edge_index), edge_index.shape)
        # print(type(A_prime), A_prime.shape)
        # print('diag_lambda val is: ', diag_lambda)
        if diag_lambda != -1:
            mask = row == col
            A_prime[mask] += diag_lambda * A_prime[mask]

        return edge_index, A_prime

    def forward(self, x, edge_index, edge_weight=None, dropout_training=True):
        """
            edge_weight (torch.tensor) : apply the pre-calculated normalized edge weights
            dropout_training (bool): swith if true means this is a training and apply dropout, otherwise won't apply dropout
        """
        self.dropout_training = dropout_training
        self.pre_embedding = x   # temp save the previous embedding for concatenation in the update

        # want to use the calculated edge_weight from the whole graph:
        if edge_weight is not None:
            return self.propagate(edge_index, x = x, norm = edge_weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype, self.diag_lambda)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x = x, norm = norm)

    def message(self, x_j, norm):
        # calculate 1) step: current_embedding <- dot(edge_weights, embeddings/feature)
        #  
        res = norm.view(-1, 1) * x_j if norm is not None else x_j
        # print('inside the update, the res is :')
        # print(type(res),res.shape)

        return res

    def update(self, aggr_out):
        # aggr_out is the current embeddings
        # step 2) concatenate(current_embedding, previous embedding)
        aggr_out = torch.cat((aggr_out, self.pre_embedding), dim = 1)
        # step 3) dropout embedding
        if self.dropout > 0:
            aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.dropout_training)
        # step 4) dot (current_embedding, weights)   return matrix of shape : (:, out_channels)
        aggr_out = torch.matmul(aggr_out, self.weight)
        # print('after dot weight, inside the update, the aggr_out is :')
        # print(type(aggr_out), aggr_out.shape)
        # step(5) 
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
