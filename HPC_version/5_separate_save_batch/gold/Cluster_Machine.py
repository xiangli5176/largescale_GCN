from collections import defaultdict
import os
import shutil
import pickle
import torch
import metis
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from itertools import chain

from utils import *

class ClusteringMachine(object):
    """
    Clustering the graph, feature set and label. Performed on the CPU side
    """
    def __init__(self, edge_index, features, label):
        """
        :param edge_index: COO format of the edge indices.
        :param features: Feature matrix (ndarray).
        :param label: label vector (ndarray).
        """
        tmp = edge_index.t().numpy().tolist()
        self.graph = nx.from_edgelist(tmp)
        self.features = features
        self.label = label
        self._set_sizes()
        self.edge_index = edge_index
        # this will get the edge weights in a complete graph
        self.get_edge_weight(self.edge_index, self.node_count)

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.node_count = self.features.shape[0]
        self.feature_count = self.features.shape[1]    # features all always in the columns
        self.label_count = len(np.unique(self.label.numpy()) )
        
    def get_edge_weight(self, edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        
        fill_value = 1 if not improved else 2
        # there are num_nodes self-loop edges added after the edge_index
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        
        row, col = edge_index   
        # row includes the starting points of the edges  (first row of edge_index)
        # col includes the ending points of the edges   (second row of edge_index)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        # row records the source nodes, which is the index we are trying to add
        # deg will record the out-degree of each node of x_i in all edges (x_i, x_j) including self_loops
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        self.edge_index_global_self_loops = edge_index
        # transfer from tensor to the numpy to construct the dict for the edge_weights
        edge_index = edge_index.t().numpy()
        normalized_edge_weight = normalized_edge_weight.numpy()
        num_edge, _ = edge_index.shape
        # this info can also be stored as matrix considering the memory, depends whether the matrix is sparse or not
        self.edge_weight_global_dict = {(edge_index[i][0], edge_index[i][1]) : normalized_edge_weight[i] for i in range(num_edge)}
        
#         print('after adding self-loops, edge_index is', edge_index)
        self.edge_weight_global = [ self.edge_weight_global_dict[(edge[0], edge[1])] for edge in edge_index ]
#         print('a list of the global weights : \n', self.edge_weight_global )
    
    # 1) first use different clustering method, then split each cluster into train, test and validation nodes, split edges
    def split_cluster_nodes_edges(self, test_ratio, validation_ratio, partition_num = 2):
        """
        Decomposing the graph, partitioning the features and label, creating Torch arrays.
        """
        # to keep the edge weights of the original whole graph:
        self.train_clusters, self.sg_nodes_global = self.metis_clustering(self.graph, partition_num)
#         self.train_clusters, self.sg_nodes_global = self.random_clustering(list(self.graph.nodes()), partition_num)
        self.valid_clusters = self.train_clusters
        
        relative_test_ratio = (test_ratio) / (1 - validation_ratio)
        self.sg_subgraph = {}
        
        self.sg_model_nodes_global = {}
        self.sg_validation_nodes_global = {}
        self.sg_train_nodes_global = {}
        self.sg_test_nodes_global = {}
        
        # keep the info of each cluster:
        self.info_isolate_cluster_size = {}
        self.info_model_cluster_size = {}
        self.info_validation_cluster_size = {}
        self.info_train_cluster_size = {}
        self.info_test_cluster_size = {}
        
        for cluster in self.train_clusters:
            self.sg_model_nodes_global[cluster], self.sg_validation_nodes_global[cluster] = train_test_split(self.sg_nodes_global[cluster], test_size = validation_ratio)
            self.sg_train_nodes_global[cluster], self.sg_test_nodes_global[cluster] = train_test_split(self.sg_model_nodes_global[cluster], test_size = relative_test_ratio)
            
            # record the information of each cluster:
            self.info_isolate_cluster_size[cluster] = len(self.sg_nodes_global[cluster])
            self.info_model_cluster_size[cluster] = len(self.sg_model_nodes_global[cluster])
            self.info_validation_cluster_size[cluster] = len(self.sg_validation_nodes_global[cluster])
            
            self.info_train_cluster_size[cluster] = len(self.sg_train_nodes_global[cluster])
            self.info_test_cluster_size[cluster] = len(self.sg_test_nodes_global[cluster])
    
    
    # 2) first assign train, test, validation nodes, split edges; this is based on the assumption that the clustering is no longer that important
    def split_whole_nodes_edges_then_cluster(self, test_ratio, validation_ratio):
        """
            First create train-test splits, then split train and validation into different batch seeds
            Input:  
                1) ratio of test, validation
                2) partition number of train nodes, test nodes, validation nodes
            Output:
                1) sg_validation_nodes_global, sg_train_nodes_global, sg_test_nodes_global
        """
        relative_test_ratio = (test_ratio) / (1 - validation_ratio)
        
        # first divide the nodes for the whole graph, result will always be a list of lists 
        model_nodes_global, self.valid_nodes_global = train_test_split(list(self.graph.nodes()), test_size = validation_ratio)
        self.train_nodes_global, self.test_nodes_global = train_test_split(model_nodes_global, test_size = relative_test_ratio)
        
    # just allocate each node to arandom cluster, store the membership inside each dict
    def random_clustering(self, target_nodes, partition_num):
        """
            Random clustering the nodes.
            Input: 
                1) target_nodes: list of node 
                2) number of partition to be generated
            Output: 
                1) cluster idx
                2) membership of each node
                3) list of generated clusters, each cluster includes a bunch of global node idx
        """
        # randomly divide into two clusters
        nodes_order = [node for node in target_nodes]
        random.shuffle(nodes_order)
        n = (len(nodes_order) + partition_num - 1) // partition_num
        partition_list = [nodes_order[i * n:(i + 1) * n] for i in range(partition_num)]
#         cluster_membership = {node : i for i, node_list in enumerate(partition_list) for node in node_list}
        cluster_nodes_global = {i : node_list for i, node_list in enumerate(partition_list)}
        
        return cluster_nodes_global

    def metis_clustering(self, target_graph, partition_num):
        """
        Clustering the graph with Metis. For details see:
        Input: 
            1) target graph
            2) number of parts
        Output:
            1) index of all the clusters
            2) membership of each node
        """
        (st, parts) = metis.part_graph(target_graph, partition_num)
        clusters = list(set(parts))
        cluster_nodes_global = defaultdict(list)
        for node, cluster_id in enumerate(parts):
            cluster_nodes_global[cluster_id].append(node)
        return clusters, cluster_nodes_global


    def general_isolate_clustering(self, k):
        """
            Still find the train batch, but cannot exceed the scope of the isolated clustering
        """
        self.sg_mini_train_edges_global = {}
        self.sg_mini_train_nodes_global = {}
        
        self.sg_mini_train_nodes_local = {}
        self.sg_mini_train_edges_local = {}
        self.sg_mini_train_edge_weight_local = {}
        self.sg_mini_train_features = {}
        self.sg_mini_train_labels = {}
        
        self.neighbor = defaultdict(dict)   # keep layer nodes of each layer
        self.train_accum_neighbor = defaultdict(set)
        
        self.info_train_batch_size = {}
        self.sg_subgraph = {}
        
        for cluster in self.train_clusters:
            self.sg_subgraph[cluster] = self.graph.subgraph(self.sg_nodes_global[cluster]) # for later use of generating local neighbor
            self.neighbor[cluster] = {0 : set(self.sg_train_nodes_global[cluster])}
            for layer in range(k):
                # first accumulate last layer
                self.train_accum_neighbor[cluster] |= self.neighbor[cluster][layer]
                tmp_level = set()
                for node in self.neighbor[cluster][layer]:
                    tmp_level |= set(self.sg_subgraph[cluster].neighbors(node))    # can only get the neighbor of the clustered: sg_subgraph[cluster], never beyond it
                # add the new layer of neighbors
                self.neighbor[cluster][layer+1] = tmp_level - self.train_accum_neighbor[cluster]
#                 print('layer ' + str(layer + 1) + ' : ', self.neighbor[cluster][layer+1])
            # the most outside layer: kth layer will be added:
            self.train_accum_neighbor[cluster] |= self.neighbor[cluster][k]
            batch_subgraph = self.sg_subgraph[cluster].subgraph(self.train_accum_neighbor[cluster])
            
            
            # first select all the overlapping nodes of the train nodes
            self.sg_mini_train_edges_global[cluster] = {edge for edge in batch_subgraph.edges()}
            self.sg_mini_train_nodes_global[cluster] = sorted(node for node in batch_subgraph.nodes())
            
            
            mini_mapper = {node: i for i, node in enumerate(self.sg_mini_train_nodes_global[cluster])}
            sg_node_index_local = sorted(mini_mapper.values())
            
            self.sg_mini_train_edges_local[cluster] = \
                           [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in self.sg_mini_train_edges_global[cluster] ] + \
                           [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in self.sg_mini_train_edges_global[cluster] ] + \
                           [ [i, i] for i in sg_node_index_local ]  
            
            self.sg_mini_train_edge_weight_local[cluster] = \
                            [ self.edge_weight_global_dict[(edge[0], edge[1])] for edge in self.sg_mini_train_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(edge[1], edge[0])] for edge in self.sg_mini_train_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(i, i)] for i in self.sg_mini_train_nodes_global[cluster] ]
            
#             print('train nodes global for the cluster # ' + str(cluster), self.sg_train_nodes_global[cluster])
            self.sg_mini_train_nodes_local[cluster] = [ mini_mapper[global_idx] for global_idx in self.sg_train_nodes_global[cluster] ]
            
            self.sg_mini_train_features[cluster] = self.features[self.sg_mini_train_nodes_global[cluster],:]
            self.sg_mini_train_labels[cluster] = self.label[self.sg_mini_train_nodes_global[cluster]]
            
            # record information 
            self.info_train_batch_size[cluster] = len(self.sg_mini_train_nodes_global[cluster])
        
        # at last, out of all the cluster loop do the data transfer
        self.transfer_edges_and_nodes()
        
        for cluster in self.sg_mini_train_edges_local.keys():
            self.sg_mini_train_edges_local[cluster] = torch.LongTensor(self.sg_mini_train_edges_local[cluster]).t()
            self.sg_mini_train_edge_weight_local[cluster] = torch.FloatTensor(self.sg_mini_train_edge_weight_local[cluster])
            self.sg_mini_train_nodes_local[cluster] = torch.LongTensor(self.sg_mini_train_nodes_local[cluster])
            self.sg_mini_train_features[cluster] = torch.FloatTensor(self.sg_mini_train_features[cluster])
            self.sg_mini_train_labels[cluster] = torch.LongTensor(self.sg_mini_train_labels[cluster])
        
    # select the training nodes as the mini-batch for each cluster
    def mini_batch_sample(self, target_seed, k, frac = 1):
        """
            This function is to generate the neighbors of the seed (either train nodes or validation nodes)
            params: cluster index, number of layer k, fraction of sampling from each neighbor layer
            input: 
                1) target_seed: this is the 0 layer inside self.neighbor
            output:
                1) neighbor: nodes global idx inside each layer of the batch
                2) accum_neighbor: accumulating neighbors , i.e. the final batch nodes
        """
        accum_neighbor = defaultdict(set)
        for cluster in target_seed.keys():
            neighbor = set(target_seed[cluster])  # first layer of the neighbor nodes of each cluster
            for layer in range(k):
                # first accumulate last layer
                accum_neighbor[cluster] |= neighbor
                tmp_level = set()
                for node in neighbor:
                    tmp_level |= set(self.graph.neighbors(node))  # the key here we are using self.graph, extract neighbor from the whole graph
                # add the new layer of neighbors
                tmp_level -= accum_neighbor[cluster]
                # each layer will only contains partial nodes from the previous layer
                neighbor = set(random.sample(tmp_level, int(len(tmp_level) * frac) ) ) if 0 < frac < 1 else tmp_level
    #                 print('layer ' + str(layer + 1) + ' : ', self.neighbor[cluster][layer+1])
            # the most outside layer: kth layer will be added:
            accum_neighbor[cluster] |= neighbor
        return accum_neighbor
        
    def mini_batch_generate(self, batch_file_folder, target_seed, k, fraction = 1.0, data_type = 'train'):
        """
            create the mini-batch focused on the train nodes only, include a total of k layers of neighbors of the original training nodes
            k: number of layers of neighbors for each training node
            fraction: fraction of neighbor nodes in each layer to be considered
            Input:
                1) target_seed: global ids of the nodes for seed to generate the batch
                    usually one of (train_global, test_global_, validation_global)
            Output: all tensors which are gonna be used in the train, forward procedure
                local:
                    1) sg_mini_edges_local
                    2) self.sg_mini_train_edge_weight_local
                    3) self.sg_mini_train_nodes_local
                    4) self.sg_mini_train_features
                    5) self.sg_mini_train_labels
            
        """
        # these are currently believed to be the main memory cost, storing all overlapping batch information
        # instead we store all the information inside one list to be stored in a pickle file as out-of-core mini-batch
        
        info_batch_size = {}
                
        accum_neighbor = self.mini_batch_sample(target_seed, k, frac = fraction)
        
        for cluster in target_seed.keys():
            batch_subgraph = self.graph.subgraph(accum_neighbor[cluster])
            
             # first select all the overlapping nodes of the train nodes
            mini_nodes_global = sorted(node for node in batch_subgraph.nodes())
            
            # store the global edges
            mini_edges_global = {edge for edge in batch_subgraph.edges()}
            
            
            mini_mapper = {node: i for i, node in enumerate(mini_nodes_global)}
            
            # store local index of batch nodes
            mini_nodes_local = [ mini_mapper[global_idx] for global_idx in target_seed[cluster] ]
            
            # store local index of batch edges
            mini_edges_local = \
                           [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in mini_edges_global ] + \
                           [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in mini_edges_global ] + \
                           [ [i, i] for i in sorted(mini_mapper.values()) ]  
            
            
            # store local edge weights
            mini_edge_weight_local = \
                            [ self.edge_weight_global_dict[(edge[0], edge[1])] for edge in mini_edges_global ] + \
                            [ self.edge_weight_global_dict[(edge[1], edge[0])] for edge in mini_edges_global ] + \
                            [ self.edge_weight_global_dict[(i, i)] for i in mini_nodes_global ]
            
            
            # store local features and lables
            mini_features = self.features[mini_nodes_global,:]
            mini_labels = self.label[mini_nodes_global]
            
            # record information 
            info_batch_size[cluster] = len(mini_nodes_global)
            
            mini_nodes_local = torch.LongTensor(mini_nodes_local)
            mini_edges_local = torch.LongTensor(mini_edges_local).t()
            mini_edge_weight_local = torch.FloatTensor(mini_edge_weight_local)
            mini_features = torch.FloatTensor(mini_features)
            mini_labels = torch.LongTensor(mini_labels)
            
            minibatch_data = [mini_nodes_local, mini_edges_local, mini_edge_weight_local, mini_features, mini_labels]
            
            batch_file_type_folder = batch_file_folder + data_type + '/'
            batch_file_name = batch_file_type_folder + 'batch_' + str(cluster)
            
            os.makedirs(os.path.dirname(batch_file_name), exist_ok=True)
            with open(batch_file_name, "wb") as fp:
                pickle.dump(minibatch_data, fp)
                
        return info_batch_size
        

    def mini_batch_train_clustering(self, batch_file_folder, k, fraction = 1.0, train_batch_num = 2):
        # special extreme case, each train node in one batch seed
#         self.sg_train_nodes_global = self.random_clustering(train_nodes_global, len(train_nodes_global))
        sg_train_nodes_global = self.random_clustering(self.train_nodes_global, train_batch_num)
        
        self.info_train_batch_size = self.mini_batch_generate(batch_file_folder, sg_train_nodes_global, k, fraction = 1.0, data_type = 'train')
        self.info_train_cluster_size = {key : len(val) for key, val in sg_train_nodes_global.items()}
        
    def mini_batch_validation_clustering(self, batch_file_folder, k, fraction = 1.0, valid_batch_num = 2):
        
        sg_validation_nodes_global = self.random_clustering(self.valid_nodes_global, valid_batch_num)
        self.info_validation_batch_size = self.mini_batch_generate(batch_file_folder, sg_validation_nodes_global, k, fraction = 1.0, data_type = 'validation')
        
        self.info_validation_cluster_size = {key : len(val) for key, val in sg_validation_nodes_global.items()}
    
    def mini_batch_test_clustering(self, batch_file_folder, k, fraction = 1.0, test_batch_num = 2):
        
        sg_test_nodes_global = self.random_clustering(self.test_nodes_global, test_batch_num)
        self.info_test_batch_size = self.mini_batch_generate(sg_test_nodes_global, k, fraction = 1.0)
        self.info_test_cluster_size = {key : len(val) for key, val in sg_test_nodes_global.items()}
        
        
            