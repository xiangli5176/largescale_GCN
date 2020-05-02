import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import shutil
import copy

import pandas as pd
import seaborn as sns

from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train, wholeClusterGCNTrainer_sequence


def check_folder_exist(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

''' Draw the information about the GCN calculating batch size '''
def draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr'):
    """
        Won't call this for mini-batch with no clustering 
    """
    cluster_id = clustering_machine.train_clusters    # a list of cluster indices
    cluster_datapoints = {'cluster_id': cluster_id,  \
                          'train_batch' : [clustering_machine.info_train_batch_size[idx] for idx in cluster_id], \
                          'cluster_size' : [clustering_machine.info_isolate_cluster_size[idx] for idx in cluster_id], \
                         }
                         
    df = pd.DataFrame(data=cluster_datapoints, dtype=np.int32)
    # print(df)
    df_reshape = df.melt('cluster_id', var_name = 'clusters', value_name = 'node_num')
    
    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="cluster_id", y="node_num", hue='clusters', kind='bar', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + comments)
    g.set_xlabels("Cluster ID")
    g.set_ylabels("Number of nodes")
    
    img_name = img_path + data_name + comments
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    g.savefig(img_name, bbox_inches='tight')


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


# preprocessing the data: remove all the isolated nodes, otherwise metis won't work normally
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


 
### Different execution model
""" Each specific model run """
def No_partition_run(local_clustering_machine, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, \
                     dropout = 0.3, lr = 0.01, weight_decay = 0.01):
    """
    # the partition num: will determine the training, testing and validation data
    return: test F-1 value, validation F-1 value
    """
    clustering_machine = copy.deepcopy(local_clustering_machine)
    # the accumulating neighbor nodes only contain train nodes, no hop neighbors
    clustering_machine.mini_batch_train_clustering(0)
    # 0) train the data as a whole with no parition
    gcn_trainer = wholeClusterGCNTrainer_sequence(clustering_machine, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay)
    
#     test_F1, test_accuracy = gcn_trainer.test()
    validation_F1, validation_accuracy = gcn_trainer.validate()
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    return validation_accuracy, validation_F1, time_train_total, time_data_load, gcn_trainer


def Cluster_train_batch_run(local_clustering_machine, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, neigh_layer = 1, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5):
    """
    # the partition num: will determine the training, testing and validation data
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
    clustering_machine = copy.deepcopy(local_clustering_machine)
    # defalt to contain 1 layer of neighbors of train nodes
    clustering_machine.mini_batch_train_clustering(neigh_layer)
    
    gcn_trainer = ClusterGCNTrainer_mini_Train(clustering_machine, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num)
    
#     test_F1, test_accuracy = gcn_trainer.test()
    validation_F1, validation_accuracy = gcn_trainer.validate()
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    return validation_accuracy, validation_F1, time_train_total, time_data_load, gcn_trainer


def Isolate_clustering_run(local_clustering_machine, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, neigh_layer = 1, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01):
    """
    # the partition num: will determine the training, testing and validation data
    return: test F-1 value, validation F-1 value
    """
    clustering_machine = copy.deepcopy(local_clustering_machine)
    # defalt to contain 1 layer of neighbors of train nodes
    clustering_machine.general_isolate_clustering(neigh_layer)
    gcn_trainer = ClusterGCNTrainer_mini_Train(clustering_machine, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay)
    
#     test_F1, test_accuracy = gcn_trainer.test()
    validation_F1, validation_accuracy = gcn_trainer.validate()
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    return validation_accuracy, validation_F1, time_train_total, time_data_load, gcn_trainer

def Cluster_train_valid_batch_run(local_clustering_machine, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, neigh_layer = 1, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5):
    """
    # the partition num: will determine the training, testing and validation data
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
    clustering_machine = copy.deepcopy(local_clustering_machine)
    # defalt to contain 1 layer of neighbors of train nodes
    clustering_machine.mini_batch_train_clustering(neigh_layer)
    
    gcn_trainer = ClusterGCNTrainer_mini_Train(clustering_machine, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num)
    
#     test_F1, test_accuracy = gcn_trainer.test()
    validation_F1, validation_accuracy = gcn_trainer.batch_validate()
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    return validation_accuracy, validation_F1, time_train_total, time_data_load, gcn_trainer



''' func for Execution of each specific model '''
def store_data_multi_tests(f1_data, data_name, graph_model, img_path, comments):
    run_id = sorted(f1_data.keys())
    run_data = {'run_id': run_id}
    
    run_data.update({model_name : [f1_data[key][idx] for key in run_id] for idx, model_name in enumerate(graph_model)})
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename

def draw_data_multi_tests(pickle_filename, data_name, comments, xlabel, ylabel):
    df = pd.read_pickle(pickle_filename)
    df_reshape = df.melt('run_id', var_name = 'model', value_name = ylabel)

    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="model", y=ylabel, kind='box', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' ' + ylabel + ' ' + comments)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    img_name = pickle_filename[:-4] + '_img'
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    plt.savefig(img_name, bbox_inches='tight')

''' func for tuning hyper-parameter of the mini-batch model '''
def store_data_multi_tuning(tune_params, target, data_name, img_path, comments):
    """
        tune_params: is the tuning parameter list
        target: is the result, here should be F1-score, accuraycy, load time, train time
    """
    run_ids = sorted(target.keys())   # key is the run_id
    run_data = {'run_id': run_ids}
    # the key can be converted to string or not: i.e. str(tune_val)
    # here we keep it as integer such that we want it to follow order
    tmp = {tune_val : [target[run_id][tune_val] for run_id in run_ids] for tune_val in tune_params}  # the value is list
    run_data.update(tmp)
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename