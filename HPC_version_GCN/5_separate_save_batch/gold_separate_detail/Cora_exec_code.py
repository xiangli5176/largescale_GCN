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
def set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, neigh_layer = 1, train_frac = 1.0, train_part_num = 2):
    """
        Set the batch machine plus generate the training batches
            1) data: the target dataset data
            2) intermediate_data_folder: path to store the intermediate generated data
            3) test_ratio, validation_ratio: data split ratio
            4) neigh_layer: number of hops (layers) for the neighbor nodes 
            5) train_frac: each time including fraction of the neigbor nodes in each layer
            6) valid_part_num, train_part_num, test_part_num :  batch number for validation, train and test data correspondingly
    """
    # if we use the random assignment of the code, then filtering out the isolated data may not be necessary
#     connect_edge_index, connect_features, connect_label = filter_out_isolate(data.edge_index, data.x, data.y)
#     clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label)
    print('\n' + '=' * 100)
    print('Start to generate the clustering machine:')
    t0 = time.time()
    clustering_machine = ClusteringMachine(data.edge_index, data.x, data.y)
    batch_machine_create = time.time() - t0
    print('Batch machine creation costs a total of {0:.4f} seconds!'.format(batch_machine_create))
    
#     clustering_machine.split_cluster_nodes_edges(test_ratio, validation_ratio, partition_num = train_part_num)
    # mini-batch only: split to train test valid before clustering
    print('Start to split data into train, test, validation:')
    t1 = time.time()
    clustering_machine.split_whole_nodes_edges_then_cluster(test_ratio, validation_ratio)
    data_split_time = time.time() - t1
    print('Data splitting costs a total of {0:.4f} seconds!'.format(data_split_time))
    
    # generate mini-batches
    mini_batch_folder = intermediate_data_folder + 'mini_batch_files/'
    check_folder_exist(mini_batch_folder)  # if exist then delete
    print('Start to generate the training batches:')
    t2 = time.time()
    clustering_machine.mini_batch_train_clustering(mini_batch_folder, neigh_layer, fraction = train_frac, train_batch_num = train_part_num)
    train_batch_production_time = time.time() - t2
    print('Train batches production costs a total of {0:.4f} seconds!'.format(train_batch_production_time))
    
    # stored the clustering machine with train-batch , validation-batch 
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    check_folder_exist(clustering_file_folder)  # if exist then delete
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    os.makedirs(os.path.dirname(clustering_file_name), exist_ok=True)
    
    print('Start to store the batch machine file:')
    t3 = time.time()
    with open(clustering_file_name, "wb") as fp:
        pickle.dump(clustering_machine, fp)
    batch_machine_store_time = time.time() - t3
    print('Storing batch machine after training batches generation costs a total of {0:.4f} seconds!'.format(batch_machine_store_time))
    print('\n' + '=' * 100)
    
    return mini_batch_folder
    

def set_clustering_machine_validation_batch(intermediate_data_folder, neigh_layer = 1, valid_part_num = 1):
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
    t1 = time.time()
    mini_batch_folder = intermediate_data_folder + 'mini_batch_files/'
    # for validation , fraction has to be 1.0 so that to include the information form original graph
    clustering_machine.mini_batch_validation_clustering(mini_batch_folder, neigh_layer, fraction = 1.0, valid_batch_num = valid_part_num)
    validation_batch_production_time = time.time() - t1
    print('Validation batches production costs a total of {0:.4f} seconds!'.format(validation_batch_production_time))
    print('=' * 100)
    
    # can off-line load the clustering model with train-batch generated
#     with open(clustering_file_name, "rb") as fp:
#         clustering_machine = pickle.load(fp)

    return mini_batch_folder

def Cluster_train_valid_batch_run(mini_batch_folder, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, \
                                 valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
    # Run the mini-batch model (train and validate both in batches)
    Tuning parameters:  dropout, lr (learning rate), weight_decay: l2 regularization
    return: validation accuracy value, validation F-1 value, time_training (ms), time_data_load (ms)
    """
#     gcn_trainer_batch = ClusterGCNTrainer_mini_Train(mini_batch_folder, 2, 2, 2, 2, 2, input_layers = [16], dropout=0.3)
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(mini_batch_folder, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
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
    print('Finish train and validate the model:')
    print('=' * 100)
    time_train_total = gcn_trainer.time_train_total
    time_data_load = gcn_trainer.time_train_load_data
    return validation_accuracy, validation_F1, time_train_total, time_data_load, gcn_trainer


def Cluster_train_valid_batch_investigate(mini_batch_folder, data_name, dataset, image_path, input_layer = [16, 16], epochs=300, \
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, output_period = 10, 
                                         valid_part_num = 2, train_part_num = 2, test_part_num = 1):
    """
        *** dynamically investigate the F1 score in the middle of the training after certain period ***
        output: two dict containing F1-score and accuracy of a certain epoch index
    """
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(mini_batch_folder, dataset.num_node_features, dataset.num_classes, input_layers = input_layer, dropout = dropout)
    train_create = time.time() - t0
    print('Trainer creation costs a total of {0:.4f} seconds!'.format(train_create))
    
    print('Start train the model:')
    t1 = time.time()
    Train_period_F1, Train_period_accuracy = gcn_trainer.train_investigate_F1(epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num, \
                                                            output_period = output_period, train_batch_num = train_part_num, valid_batch_num = valid_part_num)
    train_period = time.time() - t1
    print('In-process Training costs a total of {0:.4f} seconds!'.format(train_period))
    print('Finish train and validate the model:')
    print('=' * 100)
    
    return Train_period_F1, Train_period_accuracy


def generate_train_batch(data, data_name, dataset, image_data_path, intermediate_data_path, partition_nums, layers, \
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
            mini_batch_folder = set_clustering_machine(data, intermediate_data_folder, test_ratio = 0.05, validation_ratio = 0.85, \
                                                       neigh_layer = hop_layer, train_frac = 1.0, train_part_num = partn)
            
            set_clustering_machine_validation_batch(intermediate_data_folder, neigh_layer = hop_layer, valid_part_num = valid_part_num)




if __name__ == '__main__':


    # pc version test on Cora
    from torch_geometric.datasets import Planetoid
    local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    test_folder_name = 'Test_separate_hpc/train_10%_full_neigh_random_mini_epoch_400/'

    data_name = 'Cora'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name
    intermediate_data_folder = './intermediate_data/' + data_name + '/' + test_folder_name

    partition_nums = [2]
    layers = [[32]]


    generate_train_batch(data, data_name, dataset, image_data_path, intermediate_data_folder, partition_nums, layers, \
                dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, valid_part_num = 2)


