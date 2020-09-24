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
import json

from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from utils import *
from Custom_GCNConv import Net
from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train

##################################
##### Basic Runing module ########
##################################


''' Execute the testing program '''
def set_clustering_machine(data_name, data, dataset, intermediate_data_folder, datapath = None, 
                           train_batch_num = 2):
    """
        Set the batch machine plus generate the training batches
            1) data: the target dataset data
            2) intermediate_data_folder: path to store the intermediate generated data
            3) test_ratio, validation_ratio: data split ratio
            4) train_batch_num :  batch number for train
    """
    # set the tmp file for garbage tmp files, just collect the info:
    tmp_folder = intermediate_data_folder + 'tmp/'
    check_folder_exist(tmp_folder)
    os.makedirs(os.path.dirname(tmp_folder), exist_ok=True)
    
    # Set the clustering information storing path
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    data_info_file_folder = intermediate_data_folder + 'data_info/'
    check_folder_exist(clustering_file_folder)  # if exist then delete
    check_folder_exist(data_info_file_folder)  # if exist then delete
    
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    data_info_file_name = data_info_file_folder + 'data_info_file.txt'
    os.makedirs(os.path.dirname(clustering_file_folder), exist_ok=True)
    os.makedirs(os.path.dirname(data_info_file_folder), exist_ok=True)
    

    print('\n' + '=' * 100)
    print('Start to generate the clustering machine:')
    t0 = time.time()
    
    if datapath is None:
        raw_role = None
    else:
        raw_role = json.load(open(datapath + '{}/raw/role.json'.format(data_name)))
        
    # if we use the random assignment of the code, then filtering out the isolated data may not be necessary
    connect_edge_index, connect_features, connect_label, connect_role = filter_out_isolate_normalize_feature(data.edge_index, data.x, data.y, raw_role)
    clustering_machine = ClusteringMachine(connect_edge_index, connect_features, connect_label, connect_role, tmp_folder)
    
#     clustering_machine = ClusteringMachine(data.edge_index, data.x, data.y, tmp_folder)
    batch_machine_create = time.time() - t0
    print('Batch machine creation costs a total of {0:.4f} seconds!'.format(batch_machine_create))
    
    node_count = connect_features.shape[0]
    feature_count = connect_features.shape[1]    # features all always in the columns
    edge_count = connect_edge_index.shape[1]
    print('\nEdge number: ', edge_count, '\nNode number: ', node_count, '\nFeature number: ', feature_count) 
    
    # at last output the information inside the folder:
    print_dir_content_info(tmp_folder)
    
#     clustering_machine.split_cluster_nodes_edges(test_ratio, validation_ratio, partition_num = train_batch_num)
    # mini-batch only: split to train test valid before clustering
    print('Start to split data into train, test, validation:')
    t1 = time.time()
    clustering_machine.split_whole_nodes_edges_then_cluster(train_batch_num = train_batch_num)
    data_split_time = time.time() - t1
    print('Data splitting costs a total of {0:.4f} seconds!'.format(data_split_time))
    
    print('Start to store the batch machine file:')
    t3 = time.time()
    with open(clustering_file_name, "wb") as fp:
        pickle.dump(clustering_machine, fp)
    
    # data number we requred are the feature number and the classes, note after the filtering, the node number may be smaller by removing isolated nodes
    data_info = (dataset.num_node_features, dataset.num_classes )
    with open(data_info_file_name, "wb") as fp:
        pickle.dump(data_info, fp)

    batch_machine_store_time = time.time() - t3
    print('Storing batch machine after training batches generation costs a total of {0:.4f} seconds!'.format(batch_machine_store_time))
    print('\n' + '=' * 100)
    # output the memory usage information
    info_GPU_memory_folder = intermediate_data_folder + 'info_GPU_memory/'
    output_GPU_memory_usage('Memory_use_setting_cluster.txt', info_GPU_memory_folder, comment ='after setting clustering machine: ')
    
def set_clustering_machine_train_batch(intermediate_data_folder, sample_num = 10, \
                                      batch_range = (0, 1), info_folder = 'info_train_batch/', info_file = 'train_batch_size_info.csv'):
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
    info_folder = intermediate_data_folder + info_folder
    os.makedirs(os.path.dirname(info_folder), exist_ok=True)
    t2 = time.time()
    clustering_machine.mini_batch_train_clustering(intermediate_data_folder, sample_num = sample_num, \
                                                   batch_range = batch_range, info_folder = info_folder, info_file = info_file)
    train_batch_production_time = time.time() - t2
    print('Train batches production costs a total of {0:.4f} seconds!'.format(train_batch_production_time))
    print_dir_content_info(intermediate_data_folder + 'train_batch/')
    print('=' * 100)
    # output the memory usage information
    info_GPU_memory_folder = intermediate_data_folder + 'info_GPU_memory/'
    output_GPU_memory_usage('GPU_cost_setting_train_batch.txt', info_GPU_memory_folder, comment ='after generating train batches: ')



def set_clustering_machine_batch_whole_graph(intermediate_data_folder, info_folder = 'info_batch_whole_graph/'):
    """
        Generate the test for the whole graph
    """
    clustering_file_folder = intermediate_data_folder + 'clustering/'
    clustering_file_name = clustering_file_folder + 'clustering_machine.txt'
    print('\n' + '=' * 100)
    
    t0 = time.time()
    with open(clustering_file_name, "rb") as fp:
        clustering_machine = pickle.load(fp)
    batch_machine_read = time.time() - t0
    print('Batch machine reading costs a total of {0:.4f} seconds!'.format(batch_machine_read))
    
    print('Start to generate the test whole graph:')
    info_folder = intermediate_data_folder + info_folder
    os.makedirs(os.path.dirname(info_folder), exist_ok=True)
    t1 = time.time()
    
    # create the test batch for the whole graph
    clustering_machine.whole_test_clustering(intermediate_data_folder, info_folder = info_folder + 'test_batch/')
    # create the validation batch for the whole graph
    clustering_machine.whole_validation_clustering(intermediate_data_folder, info_folder = info_folder + 'validation_batch/')
    
    test_batch_production_time = time.time() - t1
    print('Test batches production costs a total of {0:.4f} seconds!'.format(test_batch_production_time))
    print_dir_content_info(intermediate_data_folder + 'test_batch/')
    print('=' * 100)
    # output the memory usage information
    info_GPU_memory_folder = intermediate_data_folder + 'info_GPU_memory/'
    output_GPU_memory_usage('GPU_memory_cost_generate_validatoin_data.txt', info_GPU_memory_folder, comment ='after generating test for whole graph: ')
    

def set_model_eval(intermediate_data_folder, input_layer = [16, 16], dropout = 0.3, improved = False, diag_lambda = -1):
    """
    Generate the model for evaluation, the model will be blank with all defaulted parameters. later will be used to load learned model
    """
    data_info_file_folder = intermediate_data_folder + 'data_info/'
    data_info_file = data_info_file_folder + 'data_info_file.txt'
    with open(data_info_file, "rb") as fp:
        num_node_features, num_classes = pickle.load(fp)
    
    model_eval = Net(num_node_features, num_classes, input_layers = input_layer, 
                     dropout = dropout, improved = improved, diag_lambda = diag_lambda)
    model_eval = model_eval.to("cpu")
        
    model_eval_file_name = "{}model_eval.pkl".format(intermediate_data_folder)
    with open(model_eval_file_name, "wb") as fp:
        pickle.dump(model_eval, fp)
    
    
def Cluster_investigate_train(tune_model_folder, intermediate_data_folder, input_layer = [16, 16], epochs=300,
                           dropout = 0.3, lr = 0.01, weight_decay = 0.01, mini_epoch_num = 5, improved = False, diag_lambda = -1, 
                              output_period = 10, train_part_num = 2):
    """
        *** dynamically investigate the F1 score in the middle of the training after certain period ***
        output: two dict containing F1-score and accuracy of a certain epoch index
    """
    data_info_file_folder = intermediate_data_folder + 'data_info/'
    data_info_file = data_info_file_folder + 'data_info_file.txt'
    with open(data_info_file, "rb") as fp:
        num_node_features, num_classes = pickle.load(fp)
    
    print('\n' + '=' * 100)
    print('Start generate the trainer:')
    t0 = time.time()
    gcn_trainer = ClusterGCNTrainer_mini_Train(intermediate_data_folder, num_node_features, num_classes, input_layers = input_layer, 
                                               dropout = dropout, improved = improved, diag_lambda = diag_lambda)
    train_create = time.time() - t0
    print('Trainer creation costs a total of {0:.4f} seconds!'.format(train_create))
    
    
    print('Start train the model:')
    t1 = time.time()
    
    gcn_trainer.train_investigate_F1(tune_model_folder, epoch_num=epochs, learning_rate=lr, weight_decay=weight_decay, mini_epoch_num = mini_epoch_num,
                                                            output_period = output_period, train_batch_num = train_part_num)
    train_period = time.time() - t1
    print('Training costs a total of {0:.4f} seconds!'.format(train_period))
    print('-' * 80)
    info_GPU_memory_folder = intermediate_data_folder + 'info_GPU_memory/'
    output_GPU_memory_usage('Memory_use_train_batch_investigate.txt', info_GPU_memory_folder, comment ='after train batch investigate: ')
    
    return gcn_trainer.time_train_total, gcn_trainer.time_train_load_data


def Cluster_investigate_evaluation(validation_model, mini_batch_folder, eval_target = 'validation_batch/validation_batch_whole'):
    """
        Evaluation func:
        Use the validation whole batch for validation
        Use the test whole batch for test
    """
    batch_file_name = mini_batch_folder + eval_target

    with open(batch_file_name, "rb") as fp:
        minibatch_data_test = pickle.load(fp)

    with torch.no_grad():
        test_nodes, test_edges, test_features, test_target = minibatch_data_test

        # select the testing nodes predictions and real labels
        y_pred = validation_model(test_edges, test_features)[test_nodes]
        # for multi-label task, first use the sigmoid function and rounding to 0, 1.0 labels
        y_pred_tag = (torch.sigmoid(y_pred)).round()
        predictions = y_pred_tag.cpu().detach().numpy()

        targets = test_target[test_nodes].cpu().detach().numpy()

        f1 = f1_score(targets, predictions, average="micro")
    #       accuracy = accuracy_score(targets.flatten(), predictions.flatten())   # for multi-class task, here have to be flatten first
        accuracy = binary_acc(targets, predictions)    # for multi-label task
#         print("\nTest F-1 score: {:.4f}".format(score))
    return (f1, accuracy)
