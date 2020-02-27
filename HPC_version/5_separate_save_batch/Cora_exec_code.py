import copy

import os
import pickle
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
from Cluster_Trainer import ClusterGCNTrainer_mini_Train


def set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = 1, train_frac = 1.0, \
                           valid_part_num = 1, train_part_num = 2, test_part_num = 1):
    """
        Set the clustering:
            1) data: the target dataset data
            2) intermediate_data_folder: path to store the intermediate generated data
            3) test_ratio, validation_ratio: data split ratio
            4) neigh_layer: number of hops (layers) for the neighbor nodes 
            5) train_frac: each time including fraction of the neigbor nodes in each layer
            6) valid_part_num, train_part_num, test_part_num :  batch number for validation, train and test data correspondingly
    """
    connect_edge_index, connect_features, connect_label = filter_out_isolate(data.edge_index, data.x, data.y)
    clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label)
#     clustering_machine.split_cluster_nodes_edges(test_ratio, validation_ratio, partition_num = train_part_num)
    # mini-batch only: split to train test valid before clustering
    clustering_machine.split_whole_nodes_edges_then_cluster(test_ratio, validation_ratio)
    
    # generate mini-batches
    
    mini_batch_folder = intermediate_data_folder + 'mini_batch_files/'
    check_folder_exist(mini_batch_folder)  # if exist then delete
    
    clustering_machine.mini_batch_train_clustering(mini_batch_folder, neigh_layer, fraction = train_frac, train_batch_num = train_part_num)
    # for validation , fraction has to be 1.0 so that to include the information form original graph
    clustering_machine.mini_batch_validation_clustering(mini_batch_folder, neigh_layer, fraction = 1.0, valid_batch_num = valid_part_num)
    
    # stored the clustering machine with train-batch , validation-batch 
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    check_folder_exist(clustering_file_folder)  # if exist then delete
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    os.makedirs(os.path.dirname(clustering_file_name), exist_ok=True)
    with open(clustering_file_name, "wb") as fp:
        pickle.dump(clustering_machine, fp)
        
    # can off-line load the clustering model with train-batch generated
#     with open(clustering_file_name, "rb") as fp:
#         clustering_machine = pickle.load(fp)

    return mini_batch_folder


def execute_one(mini_batch_folder, image_path, repeate_time = 5, input_layer = [32], epoch_num = 300, \
                dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5, \
                valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
        return all test-F1 and validation-F1 for all four models
    """
    validation_accuracy = {}
    validation_f1 = {}
    time_total_train = {}
    time_data_load = {}
    
    # Each graph model corresponds to one function below
#     graph_model = ['batch_valid', 'train_batch', 'whole_graph', 'isolate']
    graph_model = ['batch_valid']
    for i in range(repeate_time):
        model_res = []
        
        model_res.append(Cluster_train_valid_batch_run(mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                                         dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                                      valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num)[:4])
        
#         model_res.append(Isolate_clustering_run(clustering_machine, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, neigh_layer = layer_num, frac = frac, \
#                                                         dropout = dropout, lr = lr, weight_decay = weight_decay)[:4])
        
        validation_accuracy[i], validation_f1[i], time_total_train[i], time_data_load[i] = zip(*model_res)
    return graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load


"""To test one single model for different parameter values, always use the batch validate"""
def execute_tuning(tune_params, mini_batch_folder, image_path, repeate_time = 7, input_layer = [32], epoch_num = 400, \
                  dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, \
                  valid_part_num = 2, train_part_num = 2, test_part_num = 1):
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
    
    res = [{tune_val : Cluster_train_valid_batch_run(mini_batch_folder, data_name, dataset, image_path, input_layer = [tune_val], epochs=epoch_num, \
            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
            valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num)[:4] for tune_val in tune_params} for i in range(repeate_time)]
    
    for i, ref in enumerate(res):
        validation_accuracy[i] = {tune_val : res_lst[0] for tune_val, res_lst in ref.items()}
        validation_f1[i] = {tune_val : res_lst[1] for tune_val, res_lst in ref.items()}
        time_total_train[i] = {tune_val : res_lst[2] for tune_val, res_lst in ref.items()}
        time_data_load[i] = {tune_val : res_lst[3] for tune_val, res_lst in ref.items()}
        
    return validation_accuracy, validation_f1, time_total_train, time_data_load


def check_train_loss_converge(mini_batch_folder, data_name, dataset, image_path,  comments, input_layer = [32, 16], epoch_num = 300, \
                              dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5, \
                               valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    # mini-batch, but valid also in batches
    a3, v3, time3, load3, Cluster_train_valid_batch_trainer = Cluster_train_valid_batch_run(mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                                                               dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                                                               valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num)
    
    draw_Cluster_train_valid_batch = draw_trainer_info(data_name, Cluster_train_valid_batch_trainer, image_path, 'train_valid_batch_' + comments)
    draw_Cluster_train_valid_batch.draw_ave_loss_per_node()


def output_train_loss(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
                      dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2):
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            print('Start checking train loss for partition num: ' + str(partn) + ' hop layer: ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            intermediate_data_folder = intermediate_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            
            # set the batch for validation and train
            mini_batch_folder = set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = hop_layer, train_frac = 1.0, \
                           valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)
            
            check_train_loss_converge(mini_batch_folder, data_name, dataset, img_path, 'part_num_' + str(partn), input_layer = GCN_layer, epoch_num = 400, \
                                     dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, 
                                     valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)
            
#             # for the large dataset and split first case, the cluster info cannot be generated
#             clustering_machine.mini_batch_train_clustering(hop_layer)
#             draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr_' + str(hop_layer) + '_hops')
            
def output_F1_score(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
                    dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2):            
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            
            # set the save path
            print('Start running for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            intermediate_data_folder = intermediate_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            
            # set the batch for validation and train
            mini_batch_folder = set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = hop_layer, train_frac = 1.0, \
                           valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)
            
            # start to run the model, train and validation 
            graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load = \
                execute_one(mini_batch_folder, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, 
                                            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                             valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)
            
            
            validation_accuracy = store_data_multi_tests(validation_accuracy, data_name, graph_model, img_path, 'test_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Accuracy')

            validation_f1 = store_data_multi_tests(validation_f1, data_name, graph_model, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'F1 score')

            time_train = store_data_multi_tests(time_total_train, data_name, graph_model, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Train Time (ms)')

            time_load = store_data_multi_tests(time_data_load, data_name, graph_model, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Load Time (ms)')

def output_train_investigate(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
                             dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, output_period = 40, valid_part_num = 2):            
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            # set the save path
            print('Start running for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            intermediate_data_folder = intermediate_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            
            # set the batch for validation and train
            mini_batch_folder = set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = hop_layer, train_frac = 1.0, \
                           valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)

            Train_peroid_f1, Train_peroid_accuracy = execute_investigate(mini_batch_folder, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, \
                                            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, output_period = output_period, \
                                            valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)
            
            Train_peroid_f1 = store_data_multi_investigate(Train_peroid_f1, data_name, 'F1_score', img_path, 'invest_batch_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(Train_peroid_f1, data_name, 'Train_process_batch_num_' + str(partn) + '_hop_' + str(hop_layer), 'epoch number', 'F1 score')

            Train_peroid_accuracy = store_data_multi_investigate(Train_peroid_accuracy, data_name, 'Accuracy', img_path, 'invest_batch_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(Train_peroid_accuracy, data_name, 'Train_process_batch_num_' + str(partn) + '_hop_' + str(hop_layer), 'epoch number', 'Accuracy')
            
            
            
def output_tune_param(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2):
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            # Set the tune parameters and name
#             tune_name = 'batch_epoch_num'
#             tune_params = [400, 200, 100, 50, 20, 10, 5, 1]
            tune_name = 'layer_neuron_num'
            tune_params = [16, 32, 64, 128]

            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/' + 'tune_' + tune_name + '/'
            intermediate_data_folder = intermediate_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/' + 'tune_' + tune_name + '/'
            print('Start tuning for tuning param: ' + tune_name + ' partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            
            # set the batch for validation and train
            mini_batch_folder = set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = hop_layer, train_frac = 1.0, \
                           valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)

            validation_accuracy, validation_f1, time_total_train, time_data_load = execute_tuning(tune_params, mini_batch_folder, img_path, repeate_time = 7, \
                                                input_layer = GCN_layer, epoch_num = 400, dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                                valid_part_num = valid_part_num, train_part_num = partn, test_part_num = 1)

            validation_accuracy = store_data_multi_tuning(tune_params,validation_accuracy, data_name, img_path, 'accuracy_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Accuracy')

            validation_f1 = store_data_multi_tuning(tune_params, validation_f1, data_name, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'F1 score')

            time_train = store_data_multi_tuning(tune_params, time_total_train, data_name, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Train Time (ms)')

            time_load = store_data_multi_tuning(tune_params, time_data_load, data_name, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'epochs_per_batch', 'Load Time (ms)')
            
            


if __name__ == '__main__':
    # set the target directory: hpc version
    # local_data_root = '~/GCN/Datasets/'
    # test_folder_name = 'full_neigh/train_10%_hop_equal_net_layer/'
    
    # data_name = 'CoraFull'
    # dataset = CoraFull(root = local_data_root + 'CoralFull')
    # data = dataset[0]
    # image_data_path = './results/' + data_name + '/' + test_folder_name
    # intermediate_data_folder = './intermediate_data/' + data_name + '/' + test_folder_name
    # partition_nums = [32]
    # layers = [[64]]

    # pc version
    from torch_geometric.datasets import Planetoid
    local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    test_folder_name = 'Test_pickle_save_batch/train_10%_full_neigh_random_mini_epoch_400/'

    data_name = 'Cora'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name
    intermediate_data_folder = './intermediate_data/' + data_name + '/' + test_folder_name

    partition_nums = [2]
    layers = [[32]]

    # check F1-score
    output_F1_score(data, data_name, dataset, image_data_path, intermediate_data_folder, partition_nums, layers, \
                dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2)


