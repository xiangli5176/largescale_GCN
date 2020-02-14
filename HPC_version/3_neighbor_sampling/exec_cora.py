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
from tqdm import tqdm_notebook as tqdm
from torch_geometric.nn import GCNConv

from utils import *
from Custom_GCNConv import Net
from Cluster_Machine import ClusteringMachine
from Cluster_Trainer import ClusterGCNTrainer_mini_Train, wholeClusterGCNTrainer_sequence

# import data
from torch_geometric.datasets import Planetoid


if __name__ == '__main__':
    # set the target directory:
    local_data_root = '/media/xiangli/storage/projects/tmpdata/'
    test_folder_name = 'py_src_run/train_10%_hop_equal_net_layer/'

    data_name = 'Cora'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]
    image_data_path = './results/' + data_name + '/' + test_folder_name

    partition_nums = [4]
    layers = [[32]]

    # check the train loss
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            clustering_machine = set_clustering_machine(data, partition_num = partn, test_ratio = 0.05, validation_ratio = 0.85)
            print('Start checking train loss for partition num: ' + str(partn) + ' hop layer: ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'
            check_train_loss_converge(clustering_machine, data_name, dataset, img_path, 'part_num_' + str(partn), input_layer = GCN_layer, epoch_num = 400, layer_num = hop_layer, \
                                    dropout = 0.1, lr = 0.0001, weight_decay = 0.1)
            clustering_machine.mini_batch_train_clustering(hop_layer)
            draw_cluster_info(clustering_machine, data_name, img_path, comments = '_cluster_node_distr_' + str(hop_layer) + '_hops')

    # perform the validation:
    for partn in partition_nums:
        for GCN_layer in layers:
            net_layer = len(GCN_layer) + 1
            hop_layer = net_layer
            clustering_machine = set_clustering_machine(data, partition_num = partn, test_ratio = 0.05, validation_ratio = 0.85)
            print('Start running for partition num: ' + str(partn) + ' hop layer ' + str(hop_layer))
            img_path = image_data_path + 'cluster_num_' + str(partn) + '/' + 'net_layer_' + str(net_layer) + '_hop_layer_' + str(hop_layer) + '/'

            validation_accuracy, validation_f1, time_total_train, time_data_load = execute_one(clustering_machine, data_name, dataset, img_path, repeate_time = 7, input_layer = GCN_layer, epoch_num = 400, layer_num = hop_layer, \
                                                                                            dropout = 0.1, lr = 0.0001, weight_decay = 0.1)
            
            validation_accuracy = store_data_multi_tests(validation_accuracy, data_name, img_path, 'test_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_accuracy, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'Accuracy')
            validation_f1 = store_data_multi_tests(validation_f1, data_name, img_path, 'validation_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(validation_f1, data_name, 'vali_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'F1 score')
            
            time_train = store_data_multi_tests(time_total_train, data_name, img_path, 'train_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_train, data_name, 'train_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'Train Time (ms)')
            time_load = store_data_multi_tests(time_data_load, data_name, img_path, 'load_time_cluster_num_' + str(partn) + '_hops_' + str(hop_layer))
            draw_data_multi_tests(time_load, data_name, 'load_time_cluster_num_' + str(partn) + '_hop_' + str(hop_layer), 'Load Time (ms)')

