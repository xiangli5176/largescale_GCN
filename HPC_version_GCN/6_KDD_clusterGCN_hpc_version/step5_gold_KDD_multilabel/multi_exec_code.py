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

########################################
###### advanced multi-run module #######
########################################


#  To test one single model for different parameter values 
def execute_tuning_train(mini_batch_folder, tune_param_name, tune_val_label, tune_val, trainer_id = 0, input_layer = [32], epoch_num = 400, \
                  dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, \
                  train_batch_num = 2):
    """
        Tune all the hyperparameters
        1) learning rate
        2) dropout
        3) layer unit number
        4) weight decay
    """
    Trainer_folder = mini_batch_folder + 'GCN_tuning/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
#     check_folder_exist(Trainer_folder)
    os.makedirs(os.path.dirname(Trainer_folder), exist_ok=True)
    
    trainer_file_name = Trainer_folder + 'GCN_trainer_' + str(trainer_id)
    
    gcn_trainer = Cluster_tune_train_run(mini_batch_folder, input_layer = input_layer, epochs=epoch_num, \
                    dropout = dropout, lr = tune_val, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                    train_batch_num = train_batch_num)
    with open(trainer_file_name, "wb") as fp:
        pickle.dump(gcn_trainer, fp)

# this tuning validation requires the image_path to store all the results as the picle file
def execute_tuning_validation(image_path, mini_batch_folder, tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    
    Trainer_folder = mini_batch_folder + 'GCN_tuning/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    trainer_file_name = Trainer_folder + 'GCN_trainer_' + str(trainer_id)
    
    print('Start to read the GCN trainer model (parameters: weights, bias):')
    t1 = time.time()
    with open(trainer_file_name, "rb") as fp:
        gcn_trainer = pickle.load(fp)
    read_trainer = (time.time() - t1) * 1000
    print('Reading the trainer costs a total of {0:.4f} seconds!'.format(read_trainer))
    # res are: validation_accuracy, validation_F1, time_train_total, time_data_load
    res = Cluster_tune_validation_run(gcn_trainer, mini_batch_folder)
    
    # store the resulting data on the disk
    test_res_folder = image_path + 'test_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    os.makedirs(os.path.dirname(test_res_folder), exist_ok=True)
    test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
    
    with open(test_res_file, "wb") as fp:
        pickle.dump(res, fp)
        
    
def summarize_tuning_res(image_path, mini_batch_folder, tune_param_name, tune_val_label_list, tune_val_list, trainer_list):
    """
        tune_val_label_list :  label of the tuning parameter value for file location
        tune_val_list  :       the real value of the tuning parameter
    """
    validation_accuracy = {}
    validation_f1 = {}
    time_total_train = {}
    time_data_load = {}
    
    res = []
    for trainer_id in trainer_list:
        ref = {}
        for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
            test_res_folder = image_path + 'test_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
            test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
            with open(test_res_file, "rb") as fp:
                ref[tune_val] = pickle.load(fp)
        res.append(ref)
    
    for i, ref in enumerate(res):
        validation_accuracy[i] = {tune_val : res_lst[0] for tune_val, res_lst in ref.items()}
        validation_f1[i] = {tune_val : res_lst[1] for tune_val, res_lst in ref.items()}
        time_total_train[i] = {tune_val : res_lst[2] for tune_val, res_lst in ref.items()}
        time_data_load[i] = {tune_val : res_lst[3] for tune_val, res_lst in ref.items()}
        
    return validation_accuracy, validation_f1, time_total_train, time_data_load


### ======== Investigate F1-score on the fly
def execute_investigate(image_path, mini_batch_folder, tune_param_name, tune_val_label, tune_val, trainer_id = 0, input_layer = [32], epoch_num = 300, \
                        dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 20, output_period = 10, \
                         train_part_num = 2):
    """
        return all test-F1 and validation-F1 for all four models
    """
    Train_peroid_f1, Train_peroid_accuracy = Cluster_train_valid_batch_investigate(mini_batch_folder, input_layer = input_layer, epochs=epoch_num, \
                                        dropout = dropout, lr = tune_val, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, output_period = output_period, \
                                                                train_part_num = train_part_num)
    
    # store the resulting data on the disk
    test_res_folder = image_path + 'tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    os.makedirs(os.path.dirname(test_res_folder), exist_ok=True)
    test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
    
    with open(test_res_file, "wb") as fp:
        pickle.dump((Train_peroid_f1, Train_peroid_accuracy), fp)
        
def summarize_investigation_res(image_path, mini_batch_folder, trainer_list):
    Train_peroid_f1 = {}
    Train_peroid_accuracy = {}
    
    for trainer_id in trainer_list:
        ref = {}
        test_res_file = image_path + 'res_trainer_' + str(trainer_id)
        with open(test_res_file, "rb") as fp:
            Train_peroid_f1[trainer_id], Train_peroid_accuracy[trainer_id] = pickle.load(fp)
        
    return Train_peroid_f1, Train_peroid_accuracy



def step0_generate_clustering_machine(data, dataset, intermediate_data_path, train_batch_num, \
                                      validation_ratio = 0.05, test_ratio = 0.85, mini_cluster_num = 16, round_num = 2):            
    print('Start running for train batch num: ' + str(train_batch_num) )

    # set the basic settings for the future batches generation
    set_clustering_machine(data, dataset, intermediate_data_path, validation_ratio = validation_ratio, test_ratio = test_ratio, \
                           train_batch_num = train_batch_num, mini_cluster_num = mini_cluster_num, round_num = round_num)

def step1_generate_train_batch(intermediate_data_path, batch_range = (0, 1), info_folder = 'info/'):            
    # set the save path
    print('Start running for train batch num: ' + str(batch_range) )
    info_file = 'train_batch_size_info_{}.csv'.format(str(batch_range))

    # generate the train batches
    set_clustering_machine_train_batch(intermediate_data_path, \
                                       batch_range = batch_range, info_folder = info_folder, info_file = info_file)

def step2_generate_validation_whole_graph(intermediate_data_path, info_folder = 'info/'):            
    info_file = 'validation_whole_graph_size_info.csv'
    # generate all the tensors from the  whole graph for validation
    set_clustering_machine_validation_whole_graph(intermediate_data_path, info_folder = info_folder, info_file = info_file)


def step30_run_tune_train_batch(intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer, \
                    trainer_id = 0, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400):            
    print('Start running training for partition num: ' + str(train_batch_num))
    # start to tune the model, run different training 
    execute_tuning_train(intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id, input_layer = GCN_layer, epoch_num = epoch_num, \
                                    dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, \
                                  train_batch_num = train_batch_num)
            
def step40_run_tune_validation_whole(image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                    trainer_id = 0): 
    # set the save path
    print('Start running training for partition num: ' + str(train_batch_num) )
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    img_path += 'tuning_parameters/'  # further subfolder for different task

    # start to validate the model, with different tuning parameters
    execute_tuning_validation(img_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)



def step31_run_investigation_train_batch(image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer, \
                    trainer_id = 0, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400, output_period = 40):            
    # start to tune the model, and investigate the performance in the middle
    print('Start running investigation training for train batch num: ' + str(train_batch_num) + '_trainer_id_' + str(trainer_id))
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    img_path += 'investigation_figures/'  # further subfolder for different task

    execute_investigate(img_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id, input_layer = GCN_layer, epoch_num = epoch_num, \
                    dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, output_period = output_period, \
                     train_part_num = train_batch_num)
            

            
def step41_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, \
                                    train_batch_num, net_layer_num, trainer_list): 
    """
        Train investigation post-processing
        Train-validation at the same time
    """
    print('Start summarizing for train batch num: ' + str(train_batch_num) )
    # set the batch for validation and train
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    img_path += 'investigation_figures/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    # start to summarize the results into images for output

    Train_peroid_f1, Train_peroid_accuracy = summarize_investigation_res(img_path, intermediate_data_path, trainer_list)

    Train_peroid_f1 = store_data_multi_investigate(Train_peroid_f1, data_name, 'F1_score', img_path, 'invest_batch_num_' + str(train_batch_num))
    draw_data_multi_tests(Train_peroid_f1, data_name, 'Train_process_batch_num_' + str(train_batch_num), 'epoch number', 'F1 score')

    Train_peroid_accuracy = store_data_multi_investigate(Train_peroid_accuracy, data_name, 'Accuracy', img_path, 'invest_batch_num_' + str(train_batch_num) )
    draw_data_multi_tests(Train_peroid_accuracy, data_name, 'Train_process_batch_num_' + str(train_batch_num), 'epoch number', 'Accuracy')

    
def step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_path, tune_param_name, tune_val_label_list, tune_val_list, \
                                    train_batch_num, net_layer_num, trainer_list): 
    
    print('Start running training for partition num: ' + str(train_batch_num))
    # set the batch for validation and train
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    img_path += 'tuning_parameters/'  # further subfolder for different task

    # start to summarize the results into images for output

    validation_accuracy, validation_f1, time_total_train, time_data_load = summarize_tuning_res(img_path, intermediate_data_path, tune_param_name, tune_val_label_list, tune_val_list, trainer_list)

    generate_tuning_raw_data_table(validation_accuracy, img_path, 'test_acc.csv', tune_param_name)
    validation_accuracy_file = store_data_multi_tuning(tune_val_list, validation_accuracy, data_name, img_path, 'accuracy_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(validation_accuracy_file, data_name, 'test_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Accuracy')

    generate_tuning_raw_data_table(validation_f1, img_path, 'test_f1.csv', tune_param_name)
    validation_f1_file = store_data_multi_tuning(tune_val_list, validation_f1, data_name, img_path, 'validation_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(validation_f1_file, data_name, 'test_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'F1 score')

    generate_tuning_raw_data_table(time_total_train, img_path, 'time_train_total.csv', tune_param_name)
    time_train_file = store_data_multi_tuning(tune_val_list, time_total_train, data_name, img_path, 'train_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(time_train_file, data_name, 'train_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Train Time (ms)')

    generate_tuning_raw_data_table(time_data_load, img_path, 'time_load_data.csv', tune_param_name)
    time_load_file = store_data_multi_tuning(tune_val_list, time_data_load, data_name, img_path, 'load_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(time_load_file, data_name, 'load_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Load Time (ms)')
 


if __name__ == '__main__':
    # current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    # data_name = current_path[-3]
    # test_folder_name = 'train_10%_full_neigh/'
    # intermediate_data_folder = './'
    # image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # # this is the parts we divide the graph
    # origin_train_batch_num = 75
    # round_num = 1
    # train_batch_num = round_num * origin_train_batch_num

    # GCN_layer = [128]
    # net_layer_num = len(GCN_layer) + 1
    # # for non-optimization: hop_layer_num == net_layer_num
    # hop_layer_num = net_layer_num

    # tune_param_name = 'learning_rate' 

    # tune_val_label_list = [4]
    # tune_val_list = [10**(-label) for label in tune_val_label_list]

    # trainer_list = list(range(7))

    # from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit
    # remote_data_root = '~/GCN/Datasets/GraphSaint/'
    # data_class = eval(data_name)
    # dataset = data_class(root = remote_data_root + data_name)
    # data = dataset[0]
    # print_data_info(data, dataset)


    # pc version test on small version of PPI
    data_name = 'PPI_small'
    test_folder_name = 'clusterGCN_logic/'
    intermediate_data_folder = './tuning_lr_template/'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name

    # this is the parts we divide the graph
    origin_train_batch_num = 4
    round_num = 2
    train_batch_num = round_num * origin_train_batch_num

    GCN_layer = [32]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'learning_rate'
    tune_val_label_list = [3, 4]
    tune_val_list = [10**(-label) for label in tune_val_label_list]

    trainer_list = list(range(2))


    from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit, PPI_small
    remote_data_root = '/home/xiangli/projects/tmpdata/GCN/GraphSaint/'
    class_data = eval(data_name)
    dataset = class_data(root = remote_data_root + data_name)
    print('number of data', len(dataset))
    data = dataset[0]
    print_data_info(data, dataset)

    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, origin_train_batch_num, validation_ratio = 0.05, test_ratio = 0.85, mini_cluster_num = 32, round_num = round_num)

    step1_generate_train_batch(intermediate_data_folder, \
                            batch_range = (0, train_batch_num), 
                            info_folder = 'info_train_batch/' )

    step2_generate_validation_whole_graph(intermediate_data_folder, info_folder = 'info_validation_whole/')

    # tuning the mini-epoch number
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step30_run_tune_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer, \
                                trainer_id = trainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400)
                
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step40_run_tune_validation_whole(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                                trainer_id = trainer_id)

    step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label_list, tune_val_list, \
                                train_batch_num, net_layer_num, trainer_list)


    # train-batch investigate
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step31_run_investigation_train_batch(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer, \
                            trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400, output_period = 40)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        step41_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, \
                                        train_batch_num, net_layer_num, trainer_list)