import math
import torch
import scipy.sparse as sp
import scipy
import numpy as np
import time

from graphsaint_cython.norm_aggr import *
from utils import *
from samplers import *

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

    def __init__(self, adj_full_norm, adj_train, role, train_params, cpu_eval = False, num_cpu_core = 1):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        """
        self.num_cpu_core = num_cpu_core
        self.use_cuda = torch.cuda.is_available()
        if cpu_eval:
            self.use_cuda = False

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        # self.adj_full_norm : torch sparse tensor
        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())
        self.adj_train = adj_train

        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        
        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        # norm_loss_test is used in full batch evaluation (without sampling). so neighbor features are simply averaged.
        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        self.norm_loss_test[self.node_train] = 1./_denom     
        self.norm_loss_test[self.node_val] = 1./_denom
        self.norm_loss_test[self.node_test] = 1./_denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        
            
        self.norm_aggr_train = np.zeros(self.adj_train.size)
        
        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()   # sum the degree of each train node, here sum along column for adjacency matrix


    def set_sampler(self, train_phases, core_par_sampler = 1, samples_per_processor = 200):
        """
            Train_phases (a dict defined in the .yml file) : usually including : end, smapler, size_subg_edge
            end:  number of total epochs to stop
            sampler: category for sampler (e.g. edge)
            size_subg_edge:  size of the subgraph in number of edges
        """
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.subgraphs_remaining_edge_index = list()
        
        self.method_sample = train_phases['sampler']   # one of the string indicators regarding sampler methods
        if self.method_sample == 'mrw':
            if 'deg_clip' in train_phases:
                _deg_clip = int(train_phases['deg_clip'])
            else:
                _deg_clip = 100000      # setting this to a large number so essentially there is no clipping in probability
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(self.adj_train, self.node_train,
                                self.size_subg_budget, train_phases['size_frontier'], _deg_clip, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root'] * train_phases['depth']
            self.graph_sampler = rw_sampling(self.adj_train, self.node_train,
                                self.size_subg_budget, int(train_phases['num_root']), int(train_phases['depth']), 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge'] * 2
            self.graph_sampler = edge_sampling(self.adj_train, self.node_train, train_phases['size_subg_edge'], 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(self.adj_train,self.node_train, self.size_subg_budget, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        elif self.method_sample == 'full_batch':
            self.size_subg_budget = self.node_train.size
            self.graph_sampler = full_batch_sampling(self.adj_train,self.node_train, self.size_subg_budget, 
                                        core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
        else:
            raise NotImplementedError

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])
        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32)

        # For edge sampler, no need to estimate norm factors, we can calculate directly.
        # However, for integrity of the framework, we decide to follow the same procedure for all samplers: 
        # 1. sample enough number of subgraphs
        # 2. estimate norm factor alpha and lambda
        tot_sampled_nodes = 0
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break
        print()
        num_subg = len(self.subgraphs_remaining_nodes)  # each subgraph nodes are stored as one list inside the self.subgraphs_remaining_nodes
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        for v in range(self.adj_train.shape[0]):
            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v+1]
            val = np.clip(self.norm_loss_train[v]/self.norm_aggr_train[i_s:i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s:i_e] = val
        
        # normalize the self.norm_loss_train:
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg/self.norm_loss_train[self.node_train]/self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

    def par_graph_sample(self, phase):
        """
           Phase: can be a string "train"
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        # create 200 subgraphs per CPU, these 200 graphs may be generated by different cores, but 200 each time, not to exceed the memory limit
        print('sampling 200 subgraphs:   time = {:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)   # add lists into the subgraphs_remaining_indptr, each list is a subgraph
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def one_batch(self, mode='train'):
        """
            self.batch_num : for train mode, create one batch and the batch number will be increased by 1
        """
        if mode in ['val','test']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])  # include all the nodes inside the graph
            adj = self.adj_full_norm
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')   # if there is no sampled subgraphs, then make one
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),\
                                 self.subgraphs_remaining_indices.pop(),\
                                 self.subgraphs_remaining_indptr.pop()),\
                                 shape=(self.size_subgraph,self.size_subgraph))
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc = self.num_cpu_core)
            adj = adj_norm(adj, deg = self.deg_train[self.node_subgraph])
            adj = _coo_scipy2torch(adj.tocoo())
            
            self.batch_num += 1          # create one batch
            
        norm_loss = self.norm_loss_test if mode in ['val','test'] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]
        # this self.node_subgraph is to select the target nodes, can be left on the CPU
        return self.node_subgraph, adj, norm_loss


    def num_training_batches(self):
        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]   # greater or equal to the number of train nodes
