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
    def __init__(self, edge_index, features, label, role = None, tmp_folder = './tmp/', info_folder = './info/'):
        """
        :param edge_index: COO format of the edge indices.
        :param features: Feature matrix (ndarray).
        :param label: label vector (ndarray).
        :tmp_folder(string): the path of the folder to contain all the clustering information files
        """
        self.features = features
        self.label = torch.tensor(label, dtype=torch.float)
        self.role = role
        
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
        
        
    def split_whole_nodes_edges_then_cluster(self, train_batch_num = 2):
        """
            Only split nodes
            First create train-test splits, then split train and validation into different batch seeds
            Input:  
                1) ratio of validation (for convergnece), test
                2) partition number of train nodes, validation nodes (test nodes will be uses as a whole graph)
            Output:
                1) self.sg_validation_nodes_global, self.sg_train_nodes_global, self.sg_test_nodes_global
        """
        self.train_batch_num = train_batch_num
        
        # first divide the nodes for the whole graph, result will always be a list of lists 
        if self.role is None:
            raise Exception("role cannot be None for this split method!")
            
        self.test_nodes_global = self.role["te"]
        train_nodes_global = self.role["tr"]
        self.validation_nodes_global = self.role["va"]
        
        self.sg_train_nodes_global = self.random_clustering(train_nodes_global, train_batch_num)
    
    def manual_split_whole_nodes_edges_then_cluster(self, validation_ratio, test_ratio, train_batch_num = 2):
        """
            Only split nodes
            First create train-test splits, then split train and validation into different batch seeds
            Input:  
                1) ratio of validation (for convergnece), test
                2) partition number of train nodes, validation nodes (test nodes will be uses as a whole graph)
            Output:
                1) self.sg_validation_nodes_global, self.sg_train_nodes_global, self.sg_test_nodes_global
        """
        self.train_batch_num = train_batch_num
        relative_validation_ratio = (validation_ratio) / (1 - test_ratio)
        
        # first divide the nodes for the whole graph, result will always be a list of lists 
        model_nodes_global, self.test_nodes_global = train_test_split(list(self.graph.nodes()), test_size = test_ratio)
        train_nodes_global, self.validation_nodes_global = train_test_split(model_nodes_global, test_size = relative_validation_ratio)
        
        self.sg_train_nodes_global = self.random_clustering(train_nodes_global, train_batch_num)
    
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

    def metis_clustering(self, target_nodes, partition_num):
        """
            Random clustering the nodes.
            Input: 
                1) target_nodes: list of node 
                2) partition_num: number of partition to be generated
            Output: 
                1) membership of each node
        """
        target_graph = self.graph.subgraph(target_nodes)
        (st, parts) = metis.part_graph(target_graph, partition_num)
        clusters = list(set(parts))
        cluster_nodes_global = defaultdict(list)
        for node, cluster_id in enumerate(parts):
            cluster_nodes_global[cluster_id].append(node)
        return cluster_nodes_global

    # for use of test on the whole graph as a whole in CPU-side memory
    def whole_batch_generate(self, batch_file_folder, test_nodes, batch_name = 'test_batch_whole'):
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
        whole_labels_local = torch.FloatTensor(whole_labels_local)
        whole_test_nodes_local = torch.LongTensor( sorted(test_nodes) )

        whole_batch_data = [whole_test_nodes_local, whole_edges_local, whole_features_local, whole_labels_local]

        batch_file_name = batch_file_folder + batch_name

        # store the batch files
        t0 = time.time()
        with open(batch_file_name, "wb") as fp:
            pickle.dump(whole_batch_data, fp)
        store_time = ((time.time() - t0) * 1000)
        print('*** Generate batch file for # {0} batch, writing the batch file costed {1:.2f} ms ***'.format("whole graph", store_time) )
    
    
    def mini_batch_neighbor_sample(self, target_seed, sample_num = 10):
        """
            This function is to generate the neighbors of the seed (either train nodes or validation nodes)
            params: cluster index, number of layer k, fraction of sampling from each neighbor layer
            input: 
                1) target_seed: this is the 0 layer inside self.neighbor (can be either train or validation data)
                2) sample_num : number of sampling neighbor nodes around each seed
            output:
                1) accum_neighbor: accumulating neighbors , i.e. the final batch nodes
        """
        accum_neighbor = defaultdict(set)
        for cluster in target_seed.keys():
            neighbor = set(target_seed[cluster])  # first layer of the neighbor nodes of each cluster
            tmp_level = set()
            for node in neighbor:
                node_neigh = set(self.graph.neighbors(node))
                tmp_level |= set(random.sample(node_neigh, sample_num)) if len(node_neigh) > sample_num else node_neigh
                # the key here we are using self.graph, extract neighbor from the whole graph
            
            accum_neighbor[cluster] |= neighbor | tmp_level
        return accum_neighbor
    
    def mini_batch_generate_tensor(self, target_batch_nodes, target_seed_nodes):
        """
            sampled_neighbor_nodes : selected neighbor nodes, to be combined with train seeds to form a train batch
            input: 
                1) target_batch_nodes: batch nodes including sampling neighbors around seed nodes 
                2) target_seed_nodes : seed nodes for forming neighbors (can be either train or validation data)
        """
        batch_subgraph = self.graph.subgraph(target_batch_nodes)
            
         # first select all the overlapping nodes of the train nodes
        mini_nodes_global = sorted(node for node in batch_subgraph.nodes())

        # store the global edges
        mini_edges_global = {edge for edge in batch_subgraph.edges()}

        # map nodes from global index to local index
        mini_mapper = {node: i for i, node in enumerate(mini_nodes_global)}

        # store local index of batch nodes
        mini_nodes_local = [ mini_mapper[global_idx] for global_idx in target_seed_nodes ]

        # store local index of batch edges
        mini_edges_local = \
                       [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in mini_edges_global ] + \
                       [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in mini_edges_global ] 

        # store local features and lables
        mini_features = self.features[mini_nodes_global,:]
        mini_labels = self.label[mini_nodes_global]
        
        # record information 
        info_batch_node_size = len(mini_nodes_global)
        info_batch_edge_size = len(mini_edges_local) + info_batch_node_size  # add the self-edges

        mini_nodes_local = torch.LongTensor(mini_nodes_local)
        mini_edges_local = torch.LongTensor(mini_edges_local).t()
        mini_features = torch.FloatTensor(mini_features)
        mini_labels = torch.FloatTensor(mini_labels)

        minibatch_data = [mini_nodes_local, mini_edges_local, mini_features, mini_labels]
        
        return minibatch_data, info_batch_node_size, info_batch_edge_size
    
    def mini_batch_generate(self, batch_file_folder, target_seed, sample_num = 10, batch_range = (0, 1)):
        """
            create the mini-batch focused on the train nodes only, include a total of k layers of neighbors of the original training nodes
            k: number of layers of neighbors for each training node
            fraction: fraction of neighbor nodes in each layer to be considered
            Input:
                1) target_seed: global ids of the nodes for seed to generate the batch (can be train or validation)
                2) sample_num : number of sampling nodes to be selected from each seed node's neighbors
            Output: all tensors which are gonna be used in the train, forward procedure
                local:
                    1) sg_mini_edges_local
                    2) self.sg_mini_train_nodes_local
                    3) self.sg_mini_train_features
                    4) self.sg_mini_train_labels
            
        """
        accum_neighbor = self.mini_batch_neighbor_sample(target_seed, sample_num = sample_num)
        batch_start, batch_end = batch_range
        
        info_batch_node_size = {}
        info_batch_edge_size = {}
        batch_start, batch_end = batch_range
        for cluster in range(batch_start, batch_end):
            # main purpose is to avoid too large size of this batch_subgraph with too many intra-edges inside
            
            minibatch_data, info_batch_node_size[cluster], info_batch_edge_size[cluster] = self.mini_batch_generate_tensor(accum_neighbor[cluster], target_seed[cluster])
            
            # store the batch files
            t0 = time.time()
            batch_file_name = batch_file_folder + 'batch_' + str(cluster)
            
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
                if isinstance(val, list):
                    wr.writerow([key+1] + val)
                else:
                    wr.writerow([key+1, val])
                    
    def mini_batch_train_clustering(self, batch_folder, sample_num = 10, batch_range = (0, 1), info_folder = './info/', info_file = 'train_batch_size_info.csv'):
        data_type = 'train'
        batch_file_folder = "{}{}_batch/".format(batch_folder, data_type)
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.info_train_batch_node_size, self.info_train_batch_edge_size  = self.mini_batch_generate(batch_file_folder, self.sg_train_nodes_global, sample_num = sample_num, batch_range = batch_range)
        self.info_train_seed_size = {key : len(val) for key, val in self.sg_train_nodes_global.items()}
        
        self.save_info_dict(self.info_train_batch_node_size, info_file, info_folder, header = 'train_batch_node_id, train_batch_node_size')
        self.save_info_dict(self.info_train_batch_edge_size, info_file, info_folder, header = 'train_batch_edge_id, train_batch_edge_size')
        self.save_info_dict(self.info_train_seed_size, info_file, info_folder, header = 'train_seed_node_id, train_seed_node_size')
        
        
    def whole_test_clustering(self, batch_folder, info_folder = './info/'):
        data_type = 'test'
        batch_file_folder = "{}{}_batch/".format(batch_folder, data_type)
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.whole_batch_generate(batch_file_folder, self.test_nodes_global, batch_name = data_type + '_batch_whole')        
        
        
    def whole_validation_clustering(self, batch_folder, info_folder = './info/'):
        data_type = 'validation'
        batch_file_folder = "{}{}_batch/".format(batch_folder, data_type)
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.whole_batch_generate(batch_file_folder, self.validation_nodes_global, batch_name = data_type + '_batch_whole')       
        