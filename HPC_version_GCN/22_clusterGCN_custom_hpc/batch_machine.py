import math
import torch
import scipy.sparse as sp
import scipy
import numpy as np
import time

from utils import *

def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    
    Torch supports sparse tensors in COO(rdinate) format, which can efficiently store and process tensors 
    for which the majority of elements are zeros.
    A sparse tensor is represented as a pair of dense tensors: a tensor of values and a 2D tensor of indices. 
    A sparse tensor can be constructed by providing these two tensors, as well as the size of the sparse tensor 
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))


class Minibatch:
    """
        This minibatch iterator iterates over nodes for supervised learning.
        Data transferred to GPU:     A  init: 1) self.adj_full_norm;  2) self.norm_loss_test;
                                     B  set_sampler:  1) self.norm_loss_train
                                     C  one_batch : 1) subgraph adjacency matrix (adj)
    """

    def __init__(self, adj_full, adj_train, role, train_params, cpu_eval = False, mode = "train", 
                 num_clusters = 128, batch_num = 32):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        """
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda = False
        
        # store all the node roles as the numpy array:
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        
        self.adj_train = adj_train
        # print("adj train type is: {}; and shape is {}".format(type(adj_train), adj_train.shape))

        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(adj_full.shape[0])
        
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        
        # instead of assign all elements of self.norm_loss_test to the same averaged denominator, separately assingment instead. 
        # does this mean there are other meaningless roles beyond: test, train and validation?
        self.norm_loss_test[self.node_train] = 1./_denom     
        self.norm_loss_test[self.node_val] = 1./_denom
        self.norm_loss_test[self.node_test] = 1./_denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
            
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()   # sum the degree of each train node, here sum along column for adjacency matrix
        
        # for train part: use the modified adjacency matrix: with inter-cluster edges broken
        if mode == "train": 
            self.adj_full, self.parts = partition_graph(adj_train, self.node_train, num_clusters)
            self.generate_norm_loss_train(num_clusters, batch_num)
            self.num_training_batches = batch_num
            self.num_mini_clusters = num_clusters
        else:
            self.adj_full = adj_full


    def generate_norm_loss_train(self, num_clusters, batch_num):
        """
            Train_phases (a dict defined in the .yml file) : usually including : end, smapler, size_subg_edge
            end:  number of total epochs to stop
            sampler: category for sampler (e.g. edge)
            size_subg_edge:  size of the subgraph in number of edges
        """
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])

        self.norm_loss_train[self.node_train] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        
        # normalize the self.norm_loss_train:
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = batch_num/self.norm_loss_train[self.node_train]/self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

        
    def generate_train_batch(self, diag_lambda=-1):
        """
        Train batch Generator: Generate the batch for multiple clusters.
        """

        block_size = self.num_mini_clusters // self.num_training_batches
        np.random.shuffle(self.parts)  # each time shuffle different mini-clusters so that the combined batches are shuffled correspondingly
        
        for _, st in enumerate(range(0, self.num_mini_clusters, block_size)):
            # recombine mini-clusters into a single batch: pt
            node_subgraph = self.parts[st]
            for pt_idx in range(st + 1, min(st + block_size, self.num_mini_clusters)):
                node_subgraph = np.concatenate((node_subgraph, self.parts[pt_idx]), axis=0)
            
            norm_loss = self.norm_loss_train[node_subgraph]
            subgraph_adj = self.adj_full[node_subgraph, :][:, node_subgraph]

            # normlize subgraph_adj locally for each isolate subgraph
            if diag_lambda == -1:
                subgraph_adj = adj_norm(subgraph_adj, deg = self.deg_train[node_subgraph])
            else:
                subgraph_adj = adj_norm_diag_enhance(subgraph_adj, deg = self.deg_train[node_subgraph], diag_lambda = diag_lambda)
            subgraph_adj = _coo_scipy2torch(subgraph_adj.tocoo())
            
            yield (node_subgraph, subgraph_adj, norm_loss)    
    
    def generate_eval_batch(self):
        """
            Generate evaluation batch for validation/test procedures, whole graph 
        """
        node_subgraph = np.arange(self.adj_full.shape[0])  # include all the nodes inside the graph
        adj_full_norm = adj_norm(self.adj_full)  # return the normalized whole graph adj matrix; optional: diag_enhanced normalization: adj_norm_diag_enhance(...)
#             adj = adj_norm_diag_enhance(self.adj_full, diag_lambda = -1)
        adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())

        return node_subgraph, adj_full_norm, self.norm_loss_test
            