from collections import defaultdict
import os
import shutil
import pickle
import time
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

###############################################
####### GCN Isolate Clustering machine ########
###############################################



class ClusteringMachine(object):
    """
    Clustering the graph, feature set and label. Performed on the CPU side
    """
    def __init__(self, edge_index, features, label, tmp_folder = './tmp/', info_folder = './info/'):
        """
        :param edge_index: COO format of the edge indices.
        :param features: Feature matrix (ndarray).
        :param label: label vector (ndarray).
        :tmp_folder(string): the path of the folder to contain all the clustering information files
        """
        self.features = features
        self.label = label
        self._set_sizes()
        self.edge_index = edge_index
        # store the information folder for memory tracing
        self.tmp_folder = tmp_folder
        self.info_folder = info_folder
        
        tmp = edge_index.t().numpy().tolist()
        self.graph = nx.from_edgelist(tmp)
        
    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.node_count = self.features.shape[0]
        self.feature_count = self.features.shape[1]    # features all always in the columns
        self.label_count = len(np.unique(self.label.numpy()) )
        
    # 1) first use different clustering method, then split each cluster into train, test and validation nodes, split edges
    def split_cluster_nodes_edges(self, validation_ratio, test_ratio, mini_cluster_num = 4, train_batch_num = 2, round_num = 2):
        """
            1) decompose the whole graph into parition_num small mini-clusters, all the mini-cluster relevant use local variables
            2) recombine the mini-clusters into train_batch_num batches (self.sg_nodes_global)
            3) test is for the unseen data
            4) validation is for the cross-validation purpose
        """
        mini_cluster_nodes_global = self.metis_clustering(self.graph, mini_cluster_num)
        mini_cluster_id = list(mini_cluster_nodes_global.keys())
        
        relative_validation_ratio = (validation_ratio) / (1 - test_ratio)
        
        mini_cluster_test_nodes_global = {}
        mini_cluster_train_nodes_global = {}
        mini_cluster_validation_nodes_global = {}
        
        for cluster in mini_cluster_id:
            mini_cluster_model_nodes_global, mini_cluster_test_nodes_global[cluster] = \
                    train_test_split(mini_cluster_nodes_global[cluster], test_size = test_ratio)
            mini_cluster_train_nodes_global[cluster], mini_cluster_validation_nodes_global[cluster] = \
                    train_test_split(mini_cluster_model_nodes_global, test_size = relative_validation_ratio)
            
        #recombine_mini_cluster_for_batch:
        sg_test_nodes_global = {}
        self.sg_nodes_global = {}
        self.sg_train_nodes_global = {}
        self.sg_validation_nodes_global = {}
        # keep the info of each cluster:
        self.info_isolate_cluster_size = {}
        self.info_test_cluster_size = {}
        self.info_train_cluster_size = {}
        self.info_validation_cluster_size = {}
        # compute how many elements is inside each batch
        chunck_size = mini_cluster_num // train_batch_num
        for round_id in range(round_num):
            # first shuffle all the mini-cluster ids:
            mini_cluster_order = mini_cluster_id
            random.shuffle(mini_cluster_order)
            combine_group = [mini_cluster_order[i * chunck_size : (i + 1) * chunck_size] for i in range((len(mini_cluster_order) + chunck_size - 1) // chunck_size )]  
            for local_batch_id, group in enumerate(combine_group):
                global_batch_id = round_id * train_batch_num + local_batch_id
                self.sg_nodes_global[global_batch_id] = list(chain.from_iterable(mini_cluster_nodes_global[cluster_id] for cluster_id in group))
                sg_test_nodes_global[global_batch_id] = list(chain.from_iterable(mini_cluster_test_nodes_global[cluster_id] for cluster_id in group))
                self.sg_train_nodes_global[global_batch_id] = list(chain.from_iterable(mini_cluster_train_nodes_global[cluster_id] for cluster_id in group))
                self.sg_validation_nodes_global[global_batch_id] = list(chain.from_iterable(mini_cluster_validation_nodes_global[cluster_id] for cluster_id in group))
        
        # accumulate all the validation nodes for validation process:
        self.test_nodes_global = list(chain.from_iterable(sg_test_nodes_global.values()))
        
        for batch in self.sg_nodes_global.keys():
            # record the information of each recombined batch:
            self.info_isolate_cluster_size[batch] = len(self.sg_nodes_global[batch])
            self.info_train_cluster_size[batch] = len(self.sg_train_nodes_global[batch])
            self.info_validation_cluster_size[batch] = len(self.sg_validation_nodes_global[batch])
    
    
    # just allocate each node to arandom cluster, store the membership inside each dict
    def random_clustering(self, target_nodes, partition_num):
        """
            Random clustering the nodes.
            Input: 
                1) target_nodes: list of node 
                2) partition_num: number of partition to be generated
            Output: 
                1) membership of each node
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
            Random clustering the nodes.
            Input: 
                1) target_nodes: list of node 
                2) partition_num: number of partition to be generated
            Output: 
                1) membership of each node
        """
        (st, parts) = metis.part_graph(target_graph, partition_num)
        clusters = list(set(parts))
        cluster_nodes_global = defaultdict(list)
        for node, cluster_id in enumerate(parts):
            cluster_nodes_global[cluster_id].append(node)
        return cluster_nodes_global

    # for use of validation on the whole graph as a whole in CPU-side memory
    def whole_batch_generate(self, batch_file_folder, test_nodes):
        """
            For use of testing the model: generate the needed tensors for testing in CPU-memory side
        """
        # store the global edges
        whole_nodes_global = sorted(self.graph.nodes())
        whole_edges_global = {edge for edge in self.graph.edges()}
        
        whole_edges_local = \
                       [ [ left, right ] for left, right in whole_edges_global ] + \
                       [ [ right, left ] for left, right in whole_edges_global ] + \
                       [ [i, i] for i in whole_nodes_global ]  
        
        # store local features and lables
        whole_features_local = self.features
        whole_labels_local = self.label

        # transform all the data to the tensor form
        whole_edges_local = torch.LongTensor(whole_edges_local).t()
        whole_features_local = torch.FloatTensor(whole_features_local)
        whole_labels_local = torch.LongTensor(whole_labels_local)
        whole_test_nodes_local = torch.LongTensor( sorted(test_nodes) )

        whole_batch_data = [whole_test_nodes_local, whole_edges_local, whole_features_local, whole_labels_local]

        batch_file_name = batch_file_folder + 'batch_whole'

        # store the batch files
        t0 = time.time()
        with open(batch_file_name, "wb") as fp:
            pickle.dump(whole_batch_data, fp)
        store_time = ((time.time() - t0) * 1000)
        print('*** Generate batch file for # {0} batch, writing the batch file costed {1:.2f} ms ***'.format("whole graph", store_time) )
    
    
    
    def mini_batch_generate(self, batch_file_folder, train_seed, validation_seed, batch_range = (0, 1)):
        """
            create the mini-batch focused on the train nodes only, include a total of k layers of neighbors of the original training nodes
            k: number of layers of neighbors for each training node
            fraction: fraction of neighbor nodes in each layer to be considered
            Input:
                1) train_seed: global ids of the nodes for seed to generate the train batch
                2) validation_seed: global ids of the nodes for seed to generate the validation batch, to investigate convergence during training
            Output: all tensors which are gonna be used in the train, forward procedure
                local:
                    1) sg_mini_edges_local
                    2) self.sg_mini_train_edge_weight_local
                    3) self.sg_mini_train_nodes_local
                    4) self.sg_mini_train_features
                    5) self.sg_mini_train_labels
            
        """
        info_batch_node_size = {}
        info_batch_edge_size = {}
        batch_start, batch_end = batch_range
        for cluster in range(batch_start, batch_end):
            batch_subgraph = self.graph.subgraph(self.sg_nodes_global[cluster])
            
             # first generate all the global ids for all nodes in one isolate batch
            mini_nodes_global = sorted(node for node in batch_subgraph.nodes())
            
            # store the global edges
            mini_edges_global = {edge for edge in batch_subgraph.edges()}
            
            # map nodes from global index to local index
            mini_mapper = {node: i for i, node in enumerate(mini_nodes_global)}
            
            # store local index of batch nodes
            mini_train_nodes_local = [ mini_mapper[global_idx] for global_idx in train_seed[cluster] ]
            
            # store local index of batch nodes
            mini_validation_nodes_local = [ mini_mapper[global_idx] for global_idx in validation_seed[cluster] ]
            
            # store local index of batch edges
            mini_edges_local = \
                           [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in mini_edges_global ] + \
                           [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in mini_edges_global ]
            
            # store local features and lables
            mini_features = self.features[mini_nodes_global,:]
            mini_labels = self.label[mini_nodes_global]
            
            # record information 
            info_batch_node_size[cluster] = len(mini_nodes_global)
            info_batch_edge_size[cluster] = len(mini_edges_local)
            
            # transfer to the tensor type
            mini_edges_local = torch.LongTensor(mini_edges_local).t()
            mini_features = torch.FloatTensor(mini_features)
            mini_labels = torch.LongTensor(mini_labels)
            mini_train_nodes_local = torch.LongTensor(mini_train_nodes_local)
            mini_validation_nodes_local = torch.LongTensor(mini_validation_nodes_local)
            
            minibatch_data = [mini_edges_local, mini_features, mini_labels, mini_train_nodes_local, mini_validation_nodes_local]
            
            batch_file_name = batch_file_folder + 'batch_' + str(cluster)
            
            # store the batch files
            t0 = time.time()
            with open(batch_file_name, "wb") as fp:
                pickle.dump(minibatch_data, fp)
            store_time = ((time.time() - t0) * 1000)
            print('*** Generate batch file for # {0:3d} batch, writing the batch file costed {1:.2f} ms ***'.format(cluster, store_time) )
#             print('writing to the path: ', batch_file_name)
            
        return info_batch_node_size, info_batch_edge_size
    
    def save_info_dict(self, data, file_name, target_folder, header = 'key, value'):
        # output the batch size information as the csv file
#         os.makedirs(os.path.dirname(target_folder), exist_ok=True)
        target_file = target_folder + file_name
        
        with open(target_file, 'a', newline='\n') as fp:
            wr = csv.writer(fp, delimiter = ',')
            fp.write('\n')
            wr.writerow(header.split(','))
            for key, val in data.items():
                wr.writerow([key+1, val])
    
    def mini_batch_train_clustering(self, batch_folder, batch_range = (0, 1), info_folder = './info/', info_file = 'train_batch_size_info.csv'):
        data_type = 'train'
        batch_file_folder = batch_folder + data_type + '/'
        check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.info_train_batch_node_size, self.info_train_batch_edge_size  = self.mini_batch_generate(batch_file_folder, self.sg_train_nodes_global, self.sg_validation_nodes_global, batch_range = batch_range)
        self.info_train_seed_size = {key : len(val) for key, val in self.sg_train_nodes_global.items()}
        
        self.save_info_dict(self.info_train_batch_node_size, info_file, info_folder, header = 'train_batch_node_id, train_batch_node_size')
        self.save_info_dict(self.info_train_batch_edge_size, info_file, info_folder, header = 'train_batch_edge_id, train_batch_edge_size')
        self.save_info_dict(self.info_train_seed_size, info_file, info_folder, header = 'train_seed_node_id, train_seed_node_size')
        
    def whole_test_clustering(self, batch_folder, info_folder = './info/', info_file = 'test_whole_size_info.csv'):
        data_type = 'test'
        batch_file_folder = batch_folder + data_type + '/'
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.whole_batch_generate(batch_file_folder, self.test_nodes_global)        
        
