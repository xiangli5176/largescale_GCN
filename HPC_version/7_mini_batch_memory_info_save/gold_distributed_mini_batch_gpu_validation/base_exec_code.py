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


''' Execute the testing program '''
def set_clustering_machine(data, dataset, image_path, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, train_batch_num = 2, valid_batch_num = 2, test_batch_num = 2):
    """
        Set the batch machine plus generate the training batches
            1) data: the target dataset data
            2) intermediate_data_folder: path to store the intermediate generated data
            3) test_ratio, validation_ratio: data split ratio
            4) neigh_layer: number of hops (layers) for the neighbor nodes 
            5) train_frac: each time including fraction of the neigbor nodes in each layer
            6) valid_part_num, train_part_num, test_part_num :  batch number for validation, train and test data correspondingly
    """
    # set the tmp file for garbage tmp files, just collect the info:
    tmp_folder = './tmp/'
    check_folder_exist(tmp_folder)
    os.makedirs(os.path.dirname(tmp_folder), exist_ok=True)
    
    # Set the clustering information storing path
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    check_folder_exist(clustering_file_folder)  # if exist then delete
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    data_info_file_name = clustering_file_folder + 'data_info_file.txt'
    os.makedirs(os.path.dirname(clustering_file_folder), exist_ok=True)
    
    # if we use the random assignment of the code, then filtering out the isolated data may not be necessary
#     connect_edge_index, connect_features, connect_label = filter_out_isolate(data.edge_index, data.x, data.y)
#     clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label)
    print('\n' + '=' * 100)
    # start to generate the edge weights
    print('Start to generate the global edge weights')
    t00 = time.time()
    node_count = data.x.shape[0]
    get_edge_weight(data.edge_index, node_count, store_path = tmp_folder)
    edge_weight_create = time.time() - t00
    print('Edge weights creation costs a total of {0:.4f} seconds!'.format(edge_weight_create))
    
    print('Start to generate the clustering machine:')
    t0 = time.time()
    clustering_machine = ClusteringMachine(data.edge_index, data.x, data.y, tmp_folder)
    batch_machine_create = time.time() - t0
    print('Batch machine creation costs a total of {0:.4f} seconds!'.format(batch_machine_create))
    
    # at last output the information inside the folder:
    print_dir_content_info(tmp_folder)
    
#     clustering_machine.split_cluster_nodes_edges(test_ratio, validation_ratio, partition_num = train_part_num)
    # mini-batch only: split to train test valid before clustering
    print('Start to split data into train, test, validation:')
    t1 = time.time()
    clustering_machine.split_whole_nodes_edges_then_cluster(test_ratio, validation_ratio, \
                                    train_batch_num = train_batch_num, valid_batch_num = valid_batch_num, test_batch_num = test_batch_num)
    data_split_time = time.time() - t1
    print('Data splitting costs a total of {0:.4f} seconds!'.format(data_split_time))
    
    print('Start to store the batch machine file:')
    t3 = time.time()
    with open(clustering_file_name, "wb") as fp:
        pickle.dump(clustering_machine, fp)
    data_info = (dataset.num_node_features, dataset.num_classes )
    with open(data_info_file_name, "wb") as fp:
        pickle.dump(data_info, fp)
    
    batch_machine_store_time = time.time() - t3
    print('Storing batch machine after training batches generation costs a total of {0:.4f} seconds!'.format(batch_machine_store_time))
    print('\n' + '=' * 100)
    # output the memory usage information
    output_GPU_memory_usage('Memory_use_setting_cluster.txt', './info_GPU_memory/', comment ='after setting clustering machine: ')
    
    
def set_clustering_machine_train_batch(image_path, intermediate_data_folder, neigh_layer = 1, train_frac = 1.0, \
                                       batch_range = (0, 1), info_folder = './info_train_batch/', info_file = 'train_batch_size_info.csv'):
    """
        Generate the train batches
    """
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    print('\n' + '=' * 100)
    
    t0 = time.time()
    with open(clustering_file_name, "rb") as fp:
        clustering_machine = pickle.load(fp)
    batch_machine_read = time.time() - t0
    print('Batch machine reading costs a total of {0:.4f} seconds!'.format(batch_machine_read))
    
#     check_folder_exist(intermediate_data_folder)  # if exist then delete
    print('Start to generate the training batches:')
    os.makedirs(os.path.dirname(info_folder), exist_ok=True)
    t2 = time.time()
    clustering_machine.mini_batch_train_clustering(intermediate_data_folder, neigh_layer, fraction = train_frac, \
                                                   batch_range = batch_range, info_folder = info_folder, info_file = info_file)
    train_batch_production_time = time.time() - t2
    print('Train batches production costs a total of {0:.4f} seconds!'.format(train_batch_production_time))
    print_dir_content_info(intermediate_data_folder + 'train/')
    print('=' * 100)
    # output the memory usage information
    output_GPU_memory_usage(info_file, './info_GPU_memory/', comment ='after generating train batches: ')

def set_clustering_machine_validation_batch(image_path, intermediate_data_folder, neigh_layer = 1, validation_frac = 1.0, \
                                            batch_range = (0, 1), info_folder = './info_validation_batch/', info_file = 'validation_batch_size_info.csv'):
    """
        Generate the validation batches
    """
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    print('\n' + '=' * 100)
    
    t0 = time.time()
    with open(clustering_file_name, "rb") as fp:
        clustering_machine = pickle.load(fp)
    batch_machine_read = time.time() - t0
    print('Batch machine reading costs a total of {0:.4f} seconds!'.format(batch_machine_read))
    
    print('Start to generate the validation batches:')
    os.makedirs(os.path.dirname(info_folder), exist_ok=True)
    t1 = time.time()
    # for validation , fraction has to be 1.0 so that to include the information form original graph
    clustering_machine.mini_batch_validation_clustering(intermediate_data_folder, neigh_layer, fraction = validation_frac, \
                                                        batch_range = batch_range, info_folder = info_folder, info_file = info_file)
    validation_batch_production_time = time.time() - t1
    print('Validation batches production costs a total of {0:.4f} seconds!'.format(validation_batch_production_time))
    print_dir_content_info(intermediate_data_folder + 'validation/')
    print('=' * 100)
    # output the memory usage information
    output_GPU_memory_usage(info_file, './info_GPU_memory/', comment ='after generating validation batches: ')

def Cluster_train_batch_run(trainer_id, intermediate_data_folder, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, \
                                 train_part_num = 2, test_part_num = 1):
    """
    # Run the mini-batch model (train and validate both in batches)
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
    
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    data_info_file = clustering_file_folder + 'data_info_file.txt'
    with open(data_info_file, "rb") as fp:
        num_node_features, num_classes = pickle.load(fp)
    
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(intermediate_data_folder, num_node_features, num_classes, input_layers = input_layer, dropout = dropout)
    train_create = time.time() - t0
    print('Trainer creation costs a total of {0:.4f} seconds!'.format(train_create))
    
    print('Start train the model:')
    t1 = time.time()
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num, train_batch_num = train_part_num)
    train_period = time.time() - t1
    print('Training costs a total of {0:.4f} seconds!'.format(train_period))
    
    print('Start to save the GCN trainer model (parameters: weights, bias):')
    trainer_file_name = intermediate_data_folder + 'GCNtrainer/GCN_trainer_' + str(trainer_id)
    t2 = time.time()
    with open(trainer_file_name, "wb") as fp:
        pickle.dump(gcn_trainer, fp)
    store_trainer = time.time() - t2
    print('Storing the trainer costs a total of {0:.4f} seconds!'.format(store_trainer))
    print('-' * 80)
    output_GPU_memory_usage('Memory_use_batch_train.txt', './info_GPU_memory/', comment ='after generating trainer and train minibatches: ')

def Cluster_valid_batch_run(trainer_id, intermediate_data_folder, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, \
                                 valid_part_num = 2):
    print('Start to read the GCN trainer model (parameters: weights, bias):')
    trainer_file_name = intermediate_data_folder + 'GCNtrainer/GCN_trainer_' + str(trainer_id)
    t1 = time.time()
    with open(trainer_file_name, "rb") as fp:
        gcn_trainer = pickle.load(fp)
    read_trainer = (time.time() - t1) * 1000
    print('Reading the trainer costs a total of {0:.4f} seconds!'.format(read_trainer))
    
    print('Start validate the model:')
    t2 = time.time()
    validation_F1, validation_accuracy = gcn_trainer.batch_validate(valid_batch_num = valid_part_num)
    validation_period = time.time() - t2
    print('Validatoin costs a total of {0:.4f} seconds!'.format(validation_period))
    print('=' * 100)
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    
    output_GPU_memory_usage('Memory_use_batch_validation.txt', './info_GPU_memory/', comment ='after validating minibatches: ')
    
    return validation_accuracy, validation_F1, time_train_total, time_data_load


def Cluster_train_valid_batch_investigate(intermediate_data_folder, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, output_period = 10, 
                                         valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
        *** dynamically investigate the F1 score in the middle of the training after certain period ***
        output: two dict containing F1-score and accuracy of a certain epoch index
    """
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    data_info_file = clustering_file_folder + 'data_info_file.txt'
    with open(data_info_file, "rb") as fp:
        num_node_features, num_classes = pickle.load(fp)
    
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(intermediate_data_folder, num_node_features, num_classes, input_layers = input_layer, dropout = dropout)
    train_create = time.time() - t0
    print('Trainer creation costs a total of {0:.4f} seconds!'.format(train_create))
    
    print('Start train the model:')
    t1 = time.time()
    Train_period_F1, Train_period_accuracy = gcn_trainer.train_investigate_F1(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num, \
                                                            output_period = output_period, train_batch_num = train_part_num, valid_batch_num = valid_part_num)
    train_period = time.time() - t1
    print('In-process Training costs a total of {0:.4f} seconds!'.format(train_period))
    print('=' * 100)
    
    output_GPU_memory_usage('Memory_use_investigate_batch_train_valid.txt', './info_GPU_memory/', comment ='after train_validation investigate batches  minibatches: ')
    return Train_period_F1, Train_period_accuracy

# for the purpose for tuning 
def Cluster_train_valid_batch_run(intermediate_data_folder, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, \
                                 valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
    # Run the mini-batch model (train and validate both in batches)
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    data_info_file = clustering_file_folder + 'data_info_file.txt'
    with open(data_info_file, "rb") as fp:
        num_node_features, num_classes = pickle.load(fp)
    
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(intermediate_data_folder, num_node_features, num_classes, input_layers = input_layer, dropout = dropout)
    train_create = time.time() - t0
    print('Trainer creation costs a total of {0:.4f} seconds!'.format(train_create))
    
    print('Start train the model:')
    t1 = time.time()
    gcn_trainer.train(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num, train_batch_num = train_part_num)
    train_period = time.time() - t1
    print('Training costs a total of {0:.4f} seconds!'.format(train_period))
    print('-' * 80)
    
    print('Start validate the model:')
    t2 = time.time()
    validation_F1, validation_accuracy = gcn_trainer.batch_validate(valid_batch_num = valid_part_num)
    validation_period = time.time() - t2
    print('Validatoin costs a total of {0:.4f} seconds!'.format(validation_period))
    print('=' * 100)
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    
    output_GPU_memory_usage('Memory_use_train_validation_together.txt', './info_GPU_memory/', comment ='after train_validation batches  minibatches together: ')
    return validation_accuracy, validation_F1, time_train_total, time_data_load



if __name__ == '__main__':


    # pc version test on Cora
    from torch_geometric.datasets import Planetoid
    local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    test_folder_name = 'train_10%_full_neigh/'

    data_name = 'Cora'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name
    # set the current folder as the intermediate data folder so that we can easily copy either clustering 
    intermediate_data_folder = './'

    partition_nums = [2]
    layers = [[32]]

