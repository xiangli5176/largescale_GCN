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
import time

from collections import defaultdict

from torch_geometric.nn import GCNConv

from utils import *
from Custom_GCNConv import Net
from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train

from base_exec_code import *

def execute_one_train(mini_batch_folder, image_path, repeate_time = 5, input_layer = [32], epoch_num = 300, \
                dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5, \
                train_part_num = 2, test_part_num = 1):
    """
        Perform one train and store the results for all trainer
    """
    Trainer_folder = mini_batch_folder + 'GCNtrainer/'
    check_folder_exist(Trainer_folder)
    os.makedirs(os.path.dirname(Trainer_folder), exist_ok=True)
#     graph_model = ['batch_valid', 'train_batch', 'whole_graph', 'isolate']
    for trainer_id in range(repeate_time):
        model_res = []
        
        Cluster_train_batch_run(trainer_id, mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                                         dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                                         train_part_num = train_part_num, test_part_num = test_part_num)
        
def execute_one_validation(mini_batch_folder, image_path, repeate_time = 5, input_layer = [32], epoch_num = 300, \
                dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5, \
                valid_part_num = 2):
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
    for trainer_id in range(repeate_time):
        model_res = []
        model_res.append(Cluster_valid_batch_run(trainer_id, mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                                         dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                                      valid_part_num = valid_part_num)[:4])
        
        validation_accuracy[trainer_id], validation_f1[trainer_id], time_total_train[trainer_id], time_data_load[trainer_id] = zip(*model_res)
    return graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load


def execute_investigate(mini_batch_folder, image_path, repeate_time = 5, input_layer = [32], epoch_num = 300, \
                        dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 5, output_period = 10, \
                         valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
        return all test-F1 and validation-F1 for all four models
    """
    Train_peroid_f1 = {}
    Train_peroid_accuracy = {}
    
    for i in range(repeate_time):
        Train_peroid_f1[i], Train_peroid_accuracy[i] = Cluster_train_valid_batch_investigate(mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
                                            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, output_period = output_period, \
                                                                    valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num)
        
    return Train_peroid_f1, Train_peroid_accuracy

"""To test one single model for different parameter values"""
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
    
    res = [{tune_val : Cluster_train_valid_batch_run(mini_batch_folder, data_name, dataset, image_path, input_layer = input_layer, epochs=epoch_num, \
            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = tune_val, \
            valid_part_num = valid_part_num, train_part_num = train_part_num, test_part_num = test_part_num) for tune_val in tune_params} for i in range(repeate_time)]
    
    for i, ref in enumerate(res):
        validation_accuracy[i] = {tune_val : res_lst[0] for tune_val, res_lst in ref.items()}
        validation_f1[i] = {tune_val : res_lst[1] for tune_val, res_lst in ref.items()}
        time_total_train[i] = {tune_val : res_lst[2] for tune_val, res_lst in ref.items()}
        time_data_load[i] = {tune_val : res_lst[3] for tune_val, res_lst in ref.items()}
        
    return validation_accuracy, validation_f1, time_total_train, time_data_load


def step3_run_train_batch(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
                    dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2):            
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            
            # set the save path
            print('Start running training for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            intermediate_data_folder = intermediate_data_path
            
            # set the batch for validation and train
            mini_batch_folder = intermediate_data_folder
            
            # start to run the model, train and validation 
            execute_one_train(mini_batch_folder, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, 
                                            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                             train_part_num = partn, test_part_num = 1)
            

def step4_run_validation_batch(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
                    dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2):            
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            
            # set the save path
            print('Start running validation for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            intermediate_data_folder = intermediate_data_path
            
            # set the batch for validation and train
            mini_batch_folder = intermediate_data_folder
            graph_model, validation_accuracy, validation_f1, time_total_train, time_data_load = \
                execute_one_validation(mini_batch_folder, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, 
                                            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                             valid_part_num = valid_part_num)
            
            
            validation_accuracy = store_data_multi_tests(validation_accuracy, data_name, graph_model, img_path, 'test_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Accuracy')

            validation_f1 = store_data_multi_tests(validation_f1, data_name, graph_model, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'F1 score')

            time_train = store_data_multi_tests(time_total_train, data_name, graph_model, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Train Time (ms)')

            time_load = store_data_multi_tests(time_data_load, data_name, graph_model, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'models', 'Load Time (ms)')


if __name__ == '__main__':


    # pc version test on Cora
    from torch_geometric.datasets import Planetoid
    local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    test_folder_name = 'test_separate_hpc/train_10%_full_neigh/'

    data_name = 'Cora'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name
    # set the current folder as the intermediate data folder so that we can easily copy either clustering 
    intermediate_data_folder = './'

    partition_nums = [2]
    layers = [[32]]


    step0_generate_clustering_machine(data, intermediate_data_folder, partition_nums, layers)

    step1_generate_train_batch(intermediate_data_folder, partition_nums, layers)

    # step2_generate_validation_batch(intermediate_data_folder, partition_nums, layers, valid_part_num = 2)

    step3_run_train_batch(data, data_name, dataset, image_data_path, intermediate_data_folder, partition_nums, layers, \
                    dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2)

    # step4_run_validation_batch(data, data_name, dataset, image_data_path, intermediate_data_folder, partition_nums, layers, \
    #                 dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2)