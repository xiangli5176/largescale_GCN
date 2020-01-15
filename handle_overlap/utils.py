import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

import pandas as pd
import seaborn as sns

class draw_trainer_info:
    
    def __init__(self, data_name, ClusterGCNTrainer, image_save_path, comments):
        self.data_name = data_name
        self.image_save_path = image_save_path
        self.comments = comments
        epoch_id = list(range(len(ClusterGCNTrainer.record_ave_training_loss)))
        self.trainer_data = {'epoch_id': epoch_id,  \
                             'ave_loss_per_node' : ClusterGCNTrainer.record_ave_training_loss \
                             }

        self.df = pd.DataFrame(data = self.trainer_data, dtype=np.float64)
    
    def draw_ave_loss_per_node(self):
        plt.clf()
        plt.figure()
        sns.set(style='whitegrid')
        g = sns.lineplot(x="epoch_id", y="ave_loss_per_node", data = self.df)
        g.set_title(self.data_name + ' Ave training loss vs epoch ' + self.comments)
        g.set(xlabel='epoch ID', ylabel='Ave training loss per node')
#         fig = g.get_figure()
        filename = self.image_save_path + self.data_name + '_train_loss_' + self.comments
        # estalbish a new directory for the target saving file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        
#     def draw_epoch_time(self):
#         plt.clf()
#         plt.figure()
#         sns.set(style='whitegrid')
#         g = sns.lineplot(x="epoch_id", y="epoch_time", data = self.df)
#         g.set_title(self.data_name + ' time cost of each epoch during training')
#         g.set(xlabel='epoch ID', ylabel='Time (ms)')
# #         fig = g.get_figure()
#         plt.savefig(self.image_save_path + self.data_name + '_epoch_time', bbox_inches='tight')

def filter_out_isolate(edge_index, features, label):
    """
    edge_index: torch.Tensor (2 by 2N) for undirected edges in COO format
    features:  torch.Tensor(N by k)  for N nodes and K features
    label: torch.Tensor  (N, )  classifying labels for N nodes
    """
    edge_index_list = edge_index.t().numpy().tolist()
    connect_graph = nx.from_edgelist(edge_index_list)
    # filter out all the isolated nodes:
    connected_nodes_idx = sorted(node for node in connect_graph.nodes())
    # if the connected nodes is less than the total graph nodes
    if len(connected_nodes_idx) < features.shape[0]:
    #     print(edge_index.shape, type(edge_index))

        mapper = {node: i for i, node in enumerate(connected_nodes_idx)}
        connect_edge_index = [ [ mapper[edge[0]], mapper[edge[1]] ] for edge in edge_index_list ] 
    #     print(len(connected_nodes_idx), connected_nodes_idx[0], connected_nodes_idx[-1])
    #     print(np.array(connect_edge_index) )
        connect_edge_index =  torch.from_numpy(np.array(connect_edge_index)).t()
    #     print(connect_edge_index.shape)

        connect_features = features[connected_nodes_idx, :]
    #     print(connect_features.shape, type(connect_features))

        connect_label = label[connected_nodes_idx]
    #     print(connect_label.shape, type(connect_label))
        return connect_edge_index, connect_features, connect_label
    else:
        return edge_index, features, label


def draw_cluster_info(data_name, image_path, clustering_machine):
    cluster_id = list(range(len(clustering_machine.info_isolate_cluster_size)))
    cluster_datapoints = {'cluster_id': cluster_id,  \
                          'overlapping' : [clustering_machine.info_overlapping_cluster_size[idx] for idx in cluster_id], \
                          'isolate' : [clustering_machine.info_isolate_cluster_size[idx] for idx in cluster_id], \
                          'train' : [clustering_machine.info_train_cluster_size[idx] for idx in cluster_id], \
                          'test' : [clustering_machine.info_test_cluster_size[idx] for idx in cluster_id], \
                          'validation' : [clustering_machine.info_validation_cluster_size[idx] for idx in cluster_id]
                         }
                         
    df = pd.DataFrame(data=cluster_datapoints, dtype=np.int32)
    # print(df)
    df_reshape = df.melt('cluster_id', var_name = 'clusters', value_name = 'node_num')
    
    # print(newdf)
    sns.set(style='whitegrid')
    g = sns.catplot(x="cluster_id", y="node_num", hue='clusters', kind='bar', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' cluster node distribution')
    g.set_xlabels("Cluster ID")
    g.set_ylabels("Number of nodes")
    g.savefig(image_path+data_name+'_cluster_node_distr', bbox_inches='tight')



def draw_isolate_cluster_info(data_name, image_path, clustering_machine):
    cluster_id = list(range(len(clustering_machine.info_isolate_cluster_size)))
    cluster_datapoints = {'cluster_id': cluster_id,  \
                          'isolate' : [clustering_machine.info_isolate_cluster_size[idx] for idx in cluster_id], \
                          'train' : [clustering_machine.info_train_cluster_size[idx] for idx in cluster_id], \
                          'test' : [clustering_machine.info_test_cluster_size[idx] for idx in cluster_id], \
                          'validation' : [clustering_machine.info_validation_cluster_size[idx] for idx in cluster_id]
                         }
                         
    df = pd.DataFrame(data=cluster_datapoints, dtype=np.int32)
    # print(df)
    df_reshape = df.melt('cluster_id', var_name = 'clusters', value_name = 'node_num')

    plt.clf()
    plt.figure()
    # print(newdf)
    sns.set(style='whitegrid')
    g = sns.catplot(x="cluster_id", y="node_num", hue='clusters', kind='bar', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' cluster node distribution')
    g.set_xlabels("Cluster ID")
    g.set_ylabels("Number of nodes")
    plt.savefig(image_path+data_name+'_cluster_node_distr', bbox_inches='tight')





def print_data_info(dataset):
    data = dataset[0]
    print('Info (attributes) of a single data instance')
    print(data, '\n number of nodes: ', data.num_nodes, '\n number of edges: ', data.num_edges, \
      '\n number of features per ndoe: ', data.num_node_features, '\n number of edge features: ', data.num_edge_features, \
      '\n number of classifying labels of dataset: ', dataset.num_classes, \
      '\n all the attributes of data: ', data.keys)