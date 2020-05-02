import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import copy

import pandas as pd
import seaborn as sns

from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train, wholeClusterGCNTrainer_sequence


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


# initialize all the clusters:  separate data
def set_clustering_machine(data, partition_num = 10, test_ratio = 0.05, validation_ratio = 0.75):
    connect_edge_index, connect_features, connect_label = filter_out_isolate(data.edge_index, data.x, data.y)
    clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label, partition_num = partition_num)
    clustering_machine.decompose(test_ratio, validation_ratio)
    return clustering_machine


''' Draw the information about the GCN calculating batch size '''
def draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr'):
    cluster_id = clustering_machine.clusters    # a list of cluster indices
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


def check_train_loss_converge(clustering_machine, data_name, dataset, image_path,  comments, input_layer = [32, 16], epoch_num = 300, layer_num = 1, dropout = 0.3, lr = 0.0001, weight_decay = 0.01):
    a0, v0, time0, load0, Cluster_train_batch_trainer = Cluster_train_batch_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                                               dropout = dropout, lr = lr, weight_decay = weight_decay)
    draw_Cluster_train_batch = draw_trainer_info(data_name, Cluster_train_batch_trainer, image_path, 'train_batch_' + comments)
    draw_Cluster_train_batch.draw_ave_loss_per_node()
    
    a1, v1, time1, load1, Isolate_clustering_trainer = Isolate_clustering_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                                             dropout = dropout, lr = lr, weight_decay = weight_decay)
    draw_Isolate_clustering = draw_trainer_info(data_name, Isolate_clustering_trainer, image_path, 'Isolate_' + comments)
    draw_Isolate_clustering.draw_ave_loss_per_node()
    
    # whole graph version, should not work for the large scale graph
    a2, v2, time2, load2, No_partition_trainer = No_partition_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                                                 dropout = dropout, lr = lr, weight_decay = weight_decay)
    draw_No_partition = draw_trainer_info(data_name, No_partition_trainer, image_path, 'whole_' + comments)
    draw_No_partition.draw_ave_loss_per_node()
    




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
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01):
    """
    # the partition num: will determine the training, testing and validation data
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
    clustering_machine = copy.deepcopy(local_clustering_machine)
    # defalt to contain 1 layer of neighbors of train nodes
    clustering_machine.mini_batch_train_clustering(neigh_layer)
    
    gcn_trainer = ClusterGCNTrainer_mini_Train(clustering_machine, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay)
    
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

''' Execute the testing program '''
def execute_one(clustering_machine, data_name, dataset, image_path, repeate_time = 5, input_layer = [32, 16], epoch_num = 300, layer_num = 1, dropout = 0.3, lr = 0.0001, weight_decay = 0.01):
    """
        return all test-F1 and validation-F1 for all four models
    """
#     test_f1 = {}
    validation_accuracy = {}
    validation_f1 = {}
    time_total_train = {}
    time_data_load = {}
    for i in range(repeate_time):
        a0, v0, time0, load0, _ = Cluster_train_batch_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                         dropout = dropout, lr = lr, weight_decay = weight_decay)
        a1, v1, time1, load1, _ = Isolate_clustering_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                        dropout = dropout, lr = lr, weight_decay = weight_decay)
        a2, v2, time2, load2, _ = No_partition_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, 
                                                  dropout = dropout, lr = lr, weight_decay = weight_decay)
    
#         test_f1[i] = [t0, t1, t2]
        validation_accuracy[i] = [a0, a1, a2]
        validation_f1[i] = [v0, v1, v2]
        time_total_train[i] = [time0, time1, time2]
        time_data_load[i] = [load0, load1, load2]
    return validation_accuracy, validation_f1, time_total_train, time_data_load

def store_data_multi_tests(f1_data, data_name, img_path, comments):
    run_id = sorted(f1_data.keys())
    run_data = {'run_id': run_id,  \
                'train_batch' : [f1_data[key][0] for key in run_id], \
                'isolate' : [f1_data[key][1] for key in run_id], \
                'whole_graph' : [f1_data[key][2] for key in run_id], \
               }
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename

def draw_data_multi_tests(pickle_filename, data_name, comments, ylabel):
    df = pd.read_pickle(pickle_filename)
    df_reshape = df.melt('run_id', var_name = 'model', value_name = ylabel)

    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="model", y=ylabel, kind='box', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' ' + ylabel + ' ' + comments)
    g.set_xlabels("models")
    g.set_ylabels(ylabel)

    img_name = pickle_filename[:-4] + '_img'
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    plt.savefig(img_name, bbox_inches='tight')

def execute_tuning(tune_params, clustering_machine, image_path, repeate_time = 5, input_layer = [32, 32], epoch_num = 300, layer_num = 1):
    """
        Tune all the hyperparameters
        1) learning rate
        2) dropout
        3) layer unit number
        4) weight decay
    """
    validation_accuracy = {}
    validation_f1 = {}
    time_total_train = {}
    time_data_load = {}
    
    res = [[Cluster_train_batch_run(clustering_machine, data_name, dataset, image_path, \
            input_layer = tune_val, epochs=epoch_num, neigh_layer = layer_num, \
            dropout = 0.3, lr = 10**(-4), weight_decay = 10**(-3))[:4] for tune_val in tune_params] for i in range(repeate_time)]
    
    for i, lst in enumerate(res):
        validation_accuracy[i] = [val[0] for val in lst]
        validation_f1[i] = [val[1] for val in lst]
        time_total_train[i] = [val[2] for val in lst]
        time_data_load[i] = [val[3] for val in lst]
        
    return validation_accuracy, validation_f1, time_total_train, time_data_load

def store_data_multi_tuning(tune_params, target, data_name, img_path, comments):
    run_ids = sorted(target.keys())
    run_data = {'run_id': run_ids}
    tmp = {str(key) : [target[run_id][i] for run_id in run_ids] for i, key in enumerate(tune_params)}
    run_data.update(tmp)
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename