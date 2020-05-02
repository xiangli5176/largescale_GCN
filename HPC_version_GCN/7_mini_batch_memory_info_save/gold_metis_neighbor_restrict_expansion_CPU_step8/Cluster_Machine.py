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

##########################################
###      Mini clustering machine       ###
##########################################



class ClusteringMachine(object):
    """
    Clustering the graph, feature set and label. Performed on the CPU side
    """
    def __init__(self, edge_index, features, label, tmp_folder = './tmp/'):
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
        
        edge_weight_file = self.tmp_folder + 'input_edge_weight_list.csv'
        self.graph = nx.read_weighted_edgelist(edge_weight_file, create_using = nx.Graph, nodetype = int)

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.node_count = self.features.shape[0]
        self.feature_count = self.features.shape[1]    # features all always in the columns
        self.label_count = len(np.unique(self.label.numpy()) )
    
    # 2) first assign train, test, validation nodes, split edges; this is based on the assumption that the clustering is no longer that important
    def split_cluster_nodes_edges(self, test_ratio, validation_ratio, train_batch_num = 2):
        """
            1) decompose the whole graph into parition_num small mini-clusters, all the mini-cluster relevant use local variables
            2) recombine the mini-clusters into train_batch_num batches (self.sg_batch_nodes_global)
        """
        self.train_batch_num = train_batch_num
        
        sg_nodes_global = self.metis_clustering(self.graph, train_batch_num)
        # to always assign the same cluster of validation and train is an extra restriction
        train_clusters = list(sg_nodes_global.keys())
        
        relative_test_ratio = (test_ratio) / (1 - validation_ratio)
        
        sg_validation_nodes_global = {}
        self.sg_train_nodes_global = {}
        self.sg_test_nodes_global = {}
        
        # keep the info of each cluster:
        self.info_isolate_cluster_size = {}
        self.info_train_cluster_size = {}
        self.info_test_cluster_size = {}
        
        for cluster in train_clusters:
            sg_model_nodes_global, sg_validation_nodes_global[cluster] = train_test_split(sg_nodes_global[cluster], test_size = validation_ratio)
            self.sg_train_nodes_global[cluster], self.sg_test_nodes_global[cluster] = train_test_split(sg_model_nodes_global, test_size = relative_test_ratio)
            
            # record the information of each cluster:
            self.info_isolate_cluster_size[cluster] = len(sg_nodes_global[cluster])
            self.info_train_cluster_size[cluster] = len(self.sg_train_nodes_global[cluster])
            self.info_test_cluster_size[cluster] = len(self.sg_test_nodes_global[cluster])
        
        self.valid_nodes_global = list(chain.from_iterable(sg_validation_nodes_global.values()))
        
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
                2) partition_num: number of partition to be generated, must be greater than 1, otherwise crash
            Output: 
                1) membership of each node
        """
        (st, parts) = metis.part_graph(target_graph, partition_num)
        cluster_nodes_global = defaultdict(list)
        for node, cluster_id in enumerate(parts):
            cluster_nodes_global[cluster_id].append(node)
        return cluster_nodes_global
        
    
    # this will sample the neighbor for use, these neighbor nodes will be combined with target_seed[cluster] to form a train batch
    def mini_batch_neighbor_sample(self, target_seed, k, frac = 1):
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
            accum_neighbor[cluster] -= set(target_seed[cluster])  # only contains neighor, does not include train seeds
        return accum_neighbor
    
    # for use of validation on the whole graph as a whole in CPU-side memory
    def whole_batch_generate(self, batch_file_folder, test_nodes):
        """
            For use of testing the model: generate the needed tensors for testing in CPU-memory side
        """
        # store the global edges
        whole_nodes_global = sorted(self.graph.nodes())
        whole_edges_global = {edge for edge in self.graph.edges()}
        
        whole_edge_weight_local = \
                        [ self.graph.edges[left, right]['weight'] for left, right in whole_edges_global ] + \
                        [ self.graph.edges[right, left]['weight'] for left, right in whole_edges_global ] + \
                        [ self.graph.edges[i, i]['weight'] for i in whole_nodes_global ]
        
        whole_edges_local = \
                       [ [ left, right ] for left, right in whole_edges_global ] + \
                       [ [ right, left ] for left, right in whole_edges_global ] + \
                       [ [i, i] for i in whole_nodes_global ]  
        
        # store local features and lables
        whole_features_local = self.features
        whole_labels_local = self.label

        # transform all the data to the tensor form
        whole_edges_local = torch.LongTensor(whole_edges_local).t()
        whole_edge_weight_local = torch.FloatTensor(whole_edge_weight_local)
        whole_features_local = torch.FloatTensor(whole_features_local)
        whole_labels_local = torch.LongTensor(whole_labels_local)
        whole_test_nodes_local = torch.LongTensor( sorted(test_nodes) )

        whole_batch_data = [whole_test_nodes_local, whole_edges_local, whole_edge_weight_local, whole_features_local, whole_labels_local]

        batch_file_name = batch_file_folder + 'batch_whole'

        # store the batch files
        t0 = time.time()
        with open(batch_file_name, "wb") as fp:
            pickle.dump(whole_batch_data, fp)
        store_time = ((time.time() - t0) * 1000)
        print('*** Generate batch file for # {0} batch, writing the batch file costed {1:.2f} ms ***'.format("whole graph", store_time) )
    
    def mini_batch_generate_tensor(self, sampled_neighbor_nodes, train_seed_nodes):
        """
            sampled_neighbor_nodes : selected neighbor nodes, to be combined with train seeds to form a train batch
        """
        train_batch_nodes = set(sampled_neighbor_nodes) | set(train_seed_nodes)
        batch_subgraph = self.graph.subgraph(train_batch_nodes)
            
         # first select all the overlapping nodes of the train nodes
        mini_nodes_global = sorted(node for node in batch_subgraph.nodes())

        # store the global edges
        mini_edges_global = {edge for edge in batch_subgraph.edges()}

        # map nodes from global index to local index
        mini_mapper = {node: i for i, node in enumerate(mini_nodes_global)}

        # store local index of batch nodes
        mini_nodes_local = [ mini_mapper[global_idx] for global_idx in train_seed_nodes ]

        # store local index of batch edges
        mini_edges_local = \
                       [ [ mini_mapper[edge[0]], mini_mapper[edge[1]] ] for edge in mini_edges_global ] + \
                       [ [ mini_mapper[edge[1]], mini_mapper[edge[0]] ] for edge in mini_edges_global ] + \
                       [ [i, i] for i in sorted(mini_mapper.values()) ]  

        mini_edge_weight_local = \
                        [ self.graph.edges[left, right]['weight'] for left, right in mini_edges_global ] + \
                        [ self.graph.edges[right, left]['weight'] for left, right in mini_edges_global ] + \
                        [ self.graph.edges[i, i]['weight'] for i in mini_nodes_global ]

        # store local features and lables
        mini_features = self.features[mini_nodes_global,:]
        mini_labels = self.label[mini_nodes_global]
        
        # store the size of each sub-divided tiny batch
        info_batch_node_size = len(mini_nodes_global)
        info_batch_edge_size = len(mini_edges_local)
        
        # transform all the data to the tensor form
        mini_nodes_local = torch.LongTensor(mini_nodes_local)
        mini_edges_local = torch.LongTensor(mini_edges_local).t()
        mini_edge_weight_local = torch.FloatTensor(mini_edge_weight_local)
        mini_features = torch.FloatTensor(mini_features)
        mini_labels = torch.LongTensor(mini_labels)

        minibatch_data = [mini_nodes_local, mini_edges_local, mini_edge_weight_local, mini_features, mini_labels]
        
        return minibatch_data, info_batch_node_size, info_batch_edge_size
    
    
    def mini_batch_divide_neighbor(self, accum_neighbor, partition_size):
        """
            accum_neighbor (set) : the total sampled neighbor of a specific cluster
            partition_size: the size of each cluster
            return  (a list of list ):  each cluster contains a certain number (partition_size) of nodes
        """
        partition_num = (len(accum_neighbor) + partition_size - 1) // partition_size
        if partition_num > 1:
            sampled_neighbor_group = self.metis_clustering(self.graph.subgraph(accum_neighbor), partition_num)
            sampled_neighbor_group = [list(val) for _, val in sampled_neighbor_group.items()]
        else:
            sampled_neighbor_group = [list(accum_neighbor)]
        return sampled_neighbor_group
    
    # main logic for generating overlapping batches from train nodes
    def mini_batch_generate(self, batch_file_folder, target_seed, partition_size, k, fraction = 1.0, batch_range = (0, 1)):
        """
            create the mini-batch focused on the train nodes only, include a total of k layers of neighbors of the original training nodes
            k: number of layers of neighbors for each training node
            fraction: fraction of neighbor nodes in each layer to be considered
            Input:
                1) target_seed: global ids of the nodes for seed to generate the batch
                    usually one of (train_global, test_global_, validation_global)
                2) partition_size: the partition size of each subset of train batch
        """
        # these are currently believed to be the main memory cost, storing all overlapping batch information
        # instead we store all the information inside one list to be stored in a pickle file as out-of-core mini-batch
        
        # this will get the edge weights in a complete graph
        
        info_batch_node_size = {}
        info_batch_edge_size = {}
                
        accum_neighbor = self.mini_batch_neighbor_sample(target_seed, k, frac = fraction)
        batch_start, batch_end = batch_range
        
        for cluster in range(batch_start, batch_end):
            # main purpose is to avoid too large size of this batch_subgraph with too many intra-edges inside
            sampled_neighbor_group = self.mini_batch_divide_neighbor(accum_neighbor[cluster], partition_size)
            train_batch_tensor_group = [self.mini_batch_generate_tensor(sampled_neighbor_nodes, target_seed[cluster]) for sampled_neighbor_nodes in sampled_neighbor_group]
            # record information 
            
            info_batch_node_size[cluster] = [ele[1] for ele in train_batch_tensor_group]
            info_batch_edge_size[cluster] = [ele[2] for ele in train_batch_tensor_group]
            minibatch_data_group = [ele[0] for ele in train_batch_tensor_group]
            # store the batch files
            t0 = time.time()
            batch_file_name = batch_file_folder + 'batch_' + str(cluster)
            # store a list of tensors
            with open(batch_file_name, "wb") as fp:
                pickle.dump(minibatch_data_group, fp)
            store_time = ((time.time() - t0) * 1000)
            print('*** Generate batch file for # {0:3d} batch, writing the batch file costed {1:.2f} ms ***'.format(cluster, store_time) )
        
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


    def mini_batch_train_clustering(self, batch_folder, partition_size, k, fraction = 1.0, batch_range = (0, 1), info_folder = './info/', info_file = 'train_batch_size_info.csv'):
        """
            batch_folder : to save the batch file to this target folder 
            info_folder : to save the batch information as csv file to this folder
            k:  number of hop of neighbor layer
            fraction:  fraction of each neighbor layer
            batch_range:   generate the batch from this range [start, end)  
        """
        data_type = 'train'
        batch_file_folder = batch_folder + data_type + '/'
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.info_train_batch_node_size, self.info_train_batch_edge_size  = self.mini_batch_generate(batch_file_folder, self.sg_train_nodes_global, partition_size, k, fraction = fraction, batch_range = batch_range)
        self.info_train_seed_size = {key : len(val) for key, val in self.sg_train_nodes_global.items()}
        
        self.save_info_dict(self.info_train_batch_node_size, info_file, info_folder, header = 'train_batch_node_id, train_batch_node_size')
        self.save_info_dict(self.info_train_batch_edge_size, info_file, info_folder, header = 'train_batch_edge_id, train_batch_edge_size')
        self.save_info_dict(self.info_train_seed_size, info_file, info_folder, header = 'train_seed_node_id, train_seed_node_size')
        
    def whole_validation_clustering(self, batch_folder, info_folder = './info/', info_file = 'validation_whole_size_info.csv'):
        data_type = 'validation'
        batch_file_folder = batch_folder + data_type + '/'
#         check_folder_exist(batch_file_folder)
        os.makedirs(os.path.dirname(batch_file_folder), exist_ok=True)
        
        self.whole_batch_generate(batch_file_folder, self.valid_nodes_global)
        
        
