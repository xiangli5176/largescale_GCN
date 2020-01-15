from collections import defaultdict
import torch
import metis
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from itertools import chain

class ClusteringMachine(object):
    """
    Clustering the graph, feature set and label. Performed on the CPU side
    """
    def __init__(self, edge_index, features, label, partition_num = 2):
        """
        :param edge_index: COO format of the edge indices.
        :param features: Feature matrix (ndarray).
        :param label: label vector (ndarray).
        """
        tmp = edge_index.t().numpy().tolist()
        self.graph = nx.from_edgelist(tmp)
        self.features = features
        self.label = label
        self.partition_num = partition_num
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
    
        
    def decompose(self, test_ratio, validation_ratio):
        """
        Decomposing the graph, partitioning the features and label, creating Torch arrays.
        """
        # to keep the edge weights of the original whole graph:
        
        self.metis_clustering()
#         self.random_clustering()
        self._set_inter_clusters()
        self.general_global_isolate_partitioning(test_ratio, validation_ratio)
        # for the wholeGCNTraniner Purpose
        self.general_accumulate_partition()
        
    def _set_inter_clusters(self):
        # independent of the clustering method:
        self.intersect_cluster = []
        for i in range(1, self.partition_num):
            tmp = [(m, n) for m, n in zip(self.clusters, self.clusters[i:])]
            self.intersect_cluster.extend(tmp)
        # initialize as the totla edges (without duplicates) all over the whole graph
        self.macro_inter_edges = set(self.graph.edges())   # a sequence of tuple to indicate edges

    # just allocate each node to arandom cluster, store the membership inside each dict
    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.partition_num)]
        # randomly divide into two clusters
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.partition_num)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}


    def general_global_isolate_partitioning(self, test_ratio, validation_ratio):
        """
        Creating data partitions and train-test splits.
        """
        self.type = 'general'
        relative_test_ratio = (test_ratio) / (1 - validation_ratio)
        self.sg_nodes_global = {}
        self.sg_edges_global = {}
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
        
        for cluster in self.clusters:
            
            self.sg_subgraph[cluster] = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            
            self.sg_nodes_global[cluster] = sorted(node for node in self.sg_subgraph[cluster].nodes())
            
            self.sg_edges_global[cluster] = {edge for edge in self.sg_subgraph[cluster].edges()}
            # substract two possible directions of edges
            self.macro_inter_edges -= set([(edge[0], edge[1]) for edge in self.sg_subgraph[cluster].edges()] +  \
                                       [(edge[1], edge[0]) for edge in self.sg_subgraph[cluster].edges()])
            
            self.sg_model_nodes_global[cluster], self.sg_validation_nodes_global[cluster] = train_test_split(self.sg_nodes_global[cluster], test_size = validation_ratio)
            self.sg_model_nodes_global[cluster] = sorted(self.sg_model_nodes_global[cluster])
            self.sg_validation_nodes_global[cluster] = sorted(self.sg_validation_nodes_global[cluster])
            
            self.sg_train_nodes_global[cluster], self.sg_test_nodes_global[cluster] = train_test_split(self.sg_model_nodes_global[cluster], test_size = relative_test_ratio)
            self.sg_train_nodes_global[cluster] = sorted(self.sg_train_nodes_global[cluster])
            self.sg_test_nodes_global[cluster] = sorted(self.sg_test_nodes_global[cluster])
            
            # record the information of each cluster:
            self.info_isolate_cluster_size[cluster] = len(self.sg_nodes_global[cluster])
            self.info_model_cluster_size[cluster] = len(self.sg_model_nodes_global[cluster])
            self.info_validation_cluster_size[cluster] = len(self.sg_validation_nodes_global[cluster])
            
            self.info_train_cluster_size[cluster] = len(self.sg_train_nodes_global[cluster])
            self.info_test_cluster_size[cluster] = len(self.sg_test_nodes_global[cluster])
    
    # accumulate all the train, test, and validation nodes 
    def general_accumulate_partition(self):
        # sum up different parts of the data
#         self.total_sg_train_nodes_global = sorted(chain.from_iterable(self.sg_train_nodes_global[cluster] for cluster in self.clusters))
#         self.total_sg_test_nodes_global = sorted(chain.from_iterable(self.sg_test_nodes_global[cluster] for cluster in self.clusters)) 
        self.total_sg_train_nodes_global = sorted(self.sg_train_nodes_global[0])
        self.total_sg_test_nodes_global = sorted(self.sg_test_nodes_global[0])
        self.total_sg_validation_nodes_global = sorted(chain.from_iterable(self.sg_validation_nodes_global[cluster] for cluster in self.clusters))
    
    
    def general_isolate_clustering(self, k):
        """
            Still find the train batch, but cannot exceed the scope of the isolated clustering
        """
        self.sg_mini_edges_global = {}
        self.sg_mini_nodes_global = {}
        
        self.sg_mini_train_nodes_local = {}
        self.sg_mini_edges_local = {}
        self.sg_mini_edge_weight_local = {}
        self.sg_mini_features = {}
        self.sg_mini_labels = {}
        
        self.neighbor = defaultdict(dict)   # keep layer nodes of each layer
        self.accum_neighbor = defaultdict(set)
        
        self.info_train_batch_size = {}
        
        for cluster in self.clusters:
            self.neighbor[cluster] = {0 : set(self.sg_train_nodes_global[cluster])}
            for layer in range(k):
                # first accumulate last layer
                self.accum_neighbor[cluster] |= self.neighbor[cluster][layer]
                tmp_level = set()
                for node in self.neighbor[cluster][layer]:
                    tmp_level |= set(self.sg_subgraph[cluster].neighbors(node))
                # add the new layer of neighbors
                self.neighbor[cluster][layer+1] = tmp_level - self.accum_neighbor[cluster]
#                 print('layer ' + str(layer + 1) + ' : ', self.neighbor[cluster][layer+1])
            # the most outside layer: kth layer will be added:
            self.accum_neighbor[cluster] |= self.neighbor[cluster][k]
            batch_subgraph = self.sg_subgraph[cluster].subgraph(self.accum_neighbor[cluster])
            
#             print('nodes for cluster ' + str(cluster) + ' are: ', sorted(node for node in batch_subgraph.nodes()))
#             print('edges for cluster ' + str(cluster) + ' are: ', {edge for edge in batch_subgraph.edges()} ) 
            
            
            # first select all the overlapping nodes of the train nodes
            self.sg_mini_edges_global[cluster] = {edge for edge in batch_subgraph.edges()}
            self.sg_mini_nodes_global[cluster] = sorted(node for node in batch_subgraph.nodes())
            
            
            mini_mapper = {node: i for i, node in enumerate(self.sg_mini_nodes_global[cluster])}
            sg_node_index_local = sorted(mini_mapper.values())
            
            self.sg_mini_edges_local[cluster] = \
                           [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in self.sg_mini_edges_global[cluster] ] + \
                           [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in self.sg_mini_edges_global[cluster] ] + \
                           [ [i, i] for i in sg_node_index_local ]  
            
            self.sg_mini_edge_weight_local[cluster] = \
                            [ self.edge_weight_global_dict[(edge[0], edge[1])] for edge in self.sg_mini_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(edge[1], edge[0])] for edge in self.sg_mini_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(i, i)] for i in self.sg_mini_nodes_global[cluster] ]
            
#             print('train nodes global for the cluster # ' + str(cluster), self.sg_train_nodes_global[cluster])
            self.sg_mini_train_nodes_local[cluster] = [ mini_mapper[global_idx] for global_idx in self.sg_train_nodes_global[cluster] ]
            
            self.sg_mini_features[cluster] = self.features[self.sg_mini_nodes_global[cluster],:]
            self.sg_mini_labels[cluster] = self.label[self.sg_mini_nodes_global[cluster]]
            
            # record information 
            self.info_train_batch_size[cluster] = len(self.sg_mini_nodes_global[cluster])
        
        # at last, out of all the cluster loop do the data transfer
        self.transfer_edges_and_nodes()
        self.mini_transfer_edges_and_nodes()
        
    def print_neighbor_list(self):
        for cluster in self.clusters:
            train_set = set(self.sg_train_nodes_global[cluster])
            for node in train_set:
                print('node ' + str(node) + ' : ', list(self.graph.neighbors(node)), type(self.graph.neighbors(node)))
                
    def get_train_neighbor(self, k):
        """
            get a collection of nodes: including k layers of neighbors together with original isolate cluster nodes
            k: number of layers of neighbors
        """
        # this self.neighbor keeps a record: in each cluster, the nodes of different layer of neighbors
        self.neighbor = defaultdict(dict)   # keep layer nodes of each layer
        self.accum_neighbor = defaultdict(set)
        for cluster in self.clusters:
            self.neighbor[cluster] = {0 : set(self.sg_train_nodes_global[cluster])}
            
            for layer in range(k):
                # first accumulate last layer
                self.accum_neighbor[cluster] |= self.neighbor[cluster][layer]
                tmp_level = set()
                for node in self.neighbor[cluster][layer]:
                    tmp_level |= set(self.graph.neighbors(node))
                # add the new layer of neighbors
                self.neighbor[cluster][layer+1] = tmp_level - self.accum_neighbor[cluster]
#                 print('layer ' + str(layer + 1) + ' : ', self.neighbor[cluster][layer+1])
            # the most outside layer: kth layer will be added:
            self.accum_neighbor[cluster] |= self.neighbor[cluster][k]
#             print('accumulating ' + str(k) + ' layers: ', self.accum_neighbor[cluster])
            # after getting the train k layer neighbor nodes, generating the graph
            batch_subgraph = self.graph.subgraph(self.accum_neighbor[cluster])
            print('nodes for cluster ' + str(cluster) + ' are: ', sorted(node for node in batch_subgraph.nodes()))
            
            print('edges for cluster ' + str(cluster) + ' are: ', {edge for edge in batch_subgraph.edges()} ) 
        
            
    # select the training nodes as the mini-batch for each cluster
    def mini_batch_train_sample(self, cluster, k, frac = 1):
        self.neighbor[cluster] = {0 : set(self.sg_train_nodes_global[cluster])}
        for layer in range(k):
            # first accumulate last layer
            self.accum_neighbor[cluster] |= self.neighbor[cluster][layer]
            tmp_level = set()
            for node in self.neighbor[cluster][layer]:
                tmp_level |= set(self.graph.neighbors(node))
            # add the new layer of neighbors
            tmp_level -= self.accum_neighbor[cluster]
            # each layer will only contains partial nodes from the previous layer
            self.neighbor[cluster][layer+1] = set(random.sample(tmp_level, int(len(tmp_level) * frac) ) ) if 0 < frac < 1 else tmp_level
#                 print('layer ' + str(layer + 1) + ' : ', self.neighbor[cluster][layer+1])
        # the most outside layer: kth layer will be added:
        self.accum_neighbor[cluster] |= self.neighbor[cluster][k]
        
    def mini_batch_train_clustering(self, k, fraction = 1.0):
        """
            create the mini-batch focused on the train nodes only
            Include a total of k layers of neighbors of the original training nodes
            k: number of layers of neighbors for each training node
        """
        self.sg_mini_edges_global = {}
        self.sg_mini_nodes_global = {}
        
        self.sg_mini_train_nodes_local = {}
        self.sg_mini_edges_local = {}
        self.sg_mini_edge_weight_local = {}
        self.sg_mini_features = {}
        self.sg_mini_labels = {}
        
        self.neighbor = defaultdict(dict)   # keep layer nodes of each layer
        self.accum_neighbor = defaultdict(set)
        
        self.info_train_batch_size = {}
        
        for cluster in self.clusters:
            self.mini_batch_train_sample(cluster, k, frac = fraction)
            batch_subgraph = self.graph.subgraph(self.accum_neighbor[cluster])
            
#             print('nodes for cluster ' + str(cluster) + ' are: ', sorted(node for node in batch_subgraph.nodes()))
#             print('edges for cluster ' + str(cluster) + ' are: ', {edge for edge in batch_subgraph.edges()} ) 
            
            # first select all the overlapping nodes of the train nodes
            self.sg_mini_edges_global[cluster] = {edge for edge in batch_subgraph.edges()}
            self.sg_mini_nodes_global[cluster] = sorted(node for node in batch_subgraph.nodes())
            
            
            mini_mapper = {node: i for i, node in enumerate(self.sg_mini_nodes_global[cluster])}
            sg_node_index_local = sorted(mini_mapper.values())
            
            self.sg_mini_edges_local[cluster] = \
                           [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in self.sg_mini_edges_global[cluster] ] + \
                           [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in self.sg_mini_edges_global[cluster] ] + \
                           [ [i, i] for i in sg_node_index_local ]  
            
            self.sg_mini_edge_weight_local[cluster] = \
                            [ self.edge_weight_global_dict[(edge[0], edge[1])] for edge in self.sg_mini_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(edge[1], edge[0])] for edge in self.sg_mini_edges_global[cluster] ] + \
                            [ self.edge_weight_global_dict[(i, i)] for i in self.sg_mini_nodes_global[cluster] ]
            
#             print('train nodes global for the cluster # ' + str(cluster), self.sg_train_nodes_global[cluster])
            self.sg_mini_train_nodes_local[cluster] = [ mini_mapper[global_idx] for global_idx in self.sg_train_nodes_global[cluster] ]
            
            self.sg_mini_features[cluster] = self.features[self.sg_mini_nodes_global[cluster],:]
            self.sg_mini_labels[cluster] = self.label[self.sg_mini_nodes_global[cluster]]
            
            # record information 
            self.info_train_batch_size[cluster] = len(self.sg_mini_nodes_global[cluster])
        
        # at last, out of all the cluster loop do the data transfer
        self.transfer_edges_and_nodes()
        self.mini_transfer_edges_and_nodes()
    
    def mini_transfer_edges_and_nodes(self):
        for cluster in self.clusters:
            self.sg_mini_edges_local[cluster] = torch.LongTensor(self.sg_mini_edges_local[cluster]).t()
            self.sg_mini_edge_weight_local[cluster] = torch.FloatTensor(self.sg_mini_edge_weight_local[cluster])
            self.sg_mini_train_nodes_local[cluster] = torch.LongTensor(self.sg_mini_train_nodes_local[cluster])
            self.sg_mini_features[cluster] = torch.FloatTensor(self.sg_mini_features[cluster])
            self.sg_mini_labels[cluster] = torch.LongTensor(self.sg_mini_labels[cluster])
        
        
    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        self.edge_weight_global = torch.FloatTensor(self.edge_weight_global)
        self.edge_index_global_self_loops = self.edge_index_global_self_loops
#         self.label = torch.LongTensor(self.label)
        for cluster in self.clusters:
            self.sg_train_nodes_global[cluster] = torch.LongTensor(self.sg_train_nodes_global[cluster])
            self.sg_test_nodes_global[cluster] = torch.LongTensor(self.sg_test_nodes_global[cluster])
            self.sg_validation_nodes_global[cluster] = torch.LongTensor(self.sg_validation_nodes_global[cluster])


