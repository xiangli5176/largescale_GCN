import copy

import os
import sys
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import random

from collections import defaultdict

from torch_geometric.nn import GCNConv

from utils import *
from Custom_GCNConv import Net
from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train, wholeClusterGCNTrainer_sequence

# import data
from torch_geometric.datasets import CoraFull

def set_clustering_machine(data, test_ratio = 0.05, validation_ratio = 0.85, valid_part_num = 8, train_part_num = 8, test_part_num = 8):
    connect_edge_index, connect_features, connect_label = filter_out_isolate(data.edge_index, data.x, data.y)
    clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label)
#     clustering_machine.split_cluster_nodes_edges(test_ratio, validation_ratio, partition_num = train_part_num)
    # mini-batch only: split to train test valid before clustering
    clustering_machine.split_whole_nodes_edges_then_cluster(test_ratio, validation_ratio, valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num)
    return clustering_machine

def execute_one(clustering_machine, image_path, repeate_time = 5, input_layer = [32, 16], epoch_num = 300, layer_num = 1, \
                dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5):
    """
        return all test-F1 and validation-F1 for all four models
    """
    validation_accuracy = {}
    validation_f1 = {}
    time_total_train = {}
    time_data_load = {}
    
    # For large dataset we have to do the batch validate
    graph_model = ['batch_valid']
    # Each graph model corresponds to one function below
#     graph_model = ['batch_valid', 'train_batch', 'whole_graph', 'isolate']
    for i in range(repeate_time):
        model_res = []
        model_res.append(Cluster_train_valid_batch_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                         dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num)[:4])
        
        
        validation_accuracy[i], validation_f1[i], time_total_train[i], time_data_load[i] = zip(*model_res)
    return graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load


"""To test one single model for different parameter values, always use the batch validate"""
def execute_tuning(tune_params, clustering_machine, image_path, repeate_time = 7, input_layer = [32, 32], epoch_num = 400, layer_num = 1, \
                  dropout = 0.1, lr = 0.0001, weight_decay = 0.1):
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
    
    res = [{tune_val : Cluster_train_valid_batch_run(clustering_machine, data_name, dataset, image_path, \
            input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = tune_val)[:4] for tune_val in tune_params} for i in range(repeate_time)]
    
    for i, ref in enumerate(res):
        validation_accuracy[i] = {tune_val : res_lst[0] for tune_val, res_lst in ref.items()}
        validation_f1[i] = {tune_val : res_lst[1] for tune_val, res_lst in ref.items()}
        time_total_train[i] = {tune_val : res_lst[2] for tune_val, res_lst in ref.items()}
        time_data_load[i] = {tune_val : res_lst[3] for tune_val, res_lst in ref.items()}
        
    return validation_accuracy, validation_f1, time_total_train, time_data_load


def check_train_loss_converge(clustering_machine, data_name, dataset, image_path,  comments, input_layer = [32, 16], epoch_num = 300, layer_num = 1, dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5):
    # mini-batch, but valid also in batches
    a3, v3, time3, load3, Cluster_train_valid_batch_trainer = Cluster_train_valid_batch_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, \
                                                                               dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num)
    draw_Cluster_train_valid_batch = draw_trainer_info(data_name, Cluster_train_valid_batch_trainer, image_path, 'train_valid_batch_' + comments)
    draw_Cluster_train_valid_batch.draw_ave_loss_per_node()



def output_train_loss(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 1, valid_part_num = 64):
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            clustering_machine = set_clustering_machine(data, test_ratio = 0.05, validation_ratio = 0.85, valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 8)
            print('Start checking train loss for partition num: ' + str(partn) + ' hop layer: ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            check_train_loss_converge(clustering_machine, data_name, dataset, img_path, 'part_num_' + str(partn), input_layer = GCN_layer, epoch_num = 400, layer_num = hop_layer, \
                                     dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num)
            
#             # for the large dataset and split first case, the cluster info cannot be generated
#             clustering_machine.mini_batch_train_clustering(hop_layer)
#             draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr_' + str(hop_layer) + '_hops')
            
def output_F1_score(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 64):            
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            # here for the more custom way of train and validation, validatoin nodes and train nodes may have different number of batches
            clustering_machine = set_clustering_machine(data, test_ratio = 0.05, validation_ratio = 0.85, valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 8)
            print('Start running for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'

            graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load = execute_one(clustering_machine, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, layer_num = hop_layer, \
                                                                dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num)

            validation_accuracy = store_data_multi_tests(validation_accuracy, data_name, graph_model, img_path, 'test_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Accuracy')

            validation_f1 = store_data_multi_tests(validation_f1, data_name, graph_model, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'F1 score')

            time_train = store_data_multi_tests(time_total_train, data_name, graph_model, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Train Time (ms)')

            time_load = store_data_multi_tests(time_data_load, data_name, graph_model, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Load Time (ms)')
            
def output_tune_param(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, valid_part_num = 64):
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            clustering_machine = set_clustering_machine(data, test_ratio = 0.05, validation_ratio = 0.85, valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 8)

            # Set the tune parameters and name
            tune_name = 'batch_epoch_num'
            tune_params = [400, 200, 100, 50, 20, 10, 5, 1]

            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/' + 'tune_' + tune_name + '/'
            print('Start tuning for tuning param: ' + tune_name + ' partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))


            validation_accuracy, validation_f1, time_total_train, time_data_load = execute_tuning(tune_params, clustering_machine, img_path, repeate_time = 7, \
                                                                                    input_layer = GCN_layer, epoch_num = 400, layer_num = hop_layer, \
                                                                                    dropout = dropout, lr = lr, weight_decay = weight_decay)

            validation_accuracy = store_data_multi_tuning(tune_params,validation_accuracy, data_name, img_path, 'accuracy_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Accuracy')

            validation_f1 = store_data_multi_tuning(tune_params, validation_f1, data_name, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'F1 score')

            time_train = store_data_multi_tuning(tune_params, time_total_train, data_name, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Train Time (ms)')

            time_load = store_data_multi_tuning(tune_params, time_data_load, data_name, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Load Time (ms)')


if __name__ == '__main__':
    # set the target directory:
    local_data_root = '~/GCN/Datasets/'
    test_folder_name = 'full_neigh/train_10%_hop_equal_net_layer/'
    
    data_name = 'CoraFull'
    dataset = CoraFull(root = local_data_root + 'CoralFull')
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name
    partition_nums = [32]
    layers = [[64]]

    # check convergence
    output_train_loss(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 1, valid_part_num = 4)

    # check F1-score
    output_F1_score(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 4)

    # tuning the mini_epoch
    output_tune_param(data, data_name, dataset, image_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, valid_part_num = 4)
