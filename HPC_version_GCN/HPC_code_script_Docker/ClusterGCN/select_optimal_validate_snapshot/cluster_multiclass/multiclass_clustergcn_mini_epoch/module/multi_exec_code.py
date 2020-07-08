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
import shutil

from collections import defaultdict

# from torch_geometric.nn import GCNConv

from utils import *

from base_exec_code import *

########################################
###### advanced multi-run module #######
########################################


### ======== Investigate test-score offline
def execute_test_tuning(image_path, mini_batch_folder, snapshot_epoch_list, 
                                 tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    """
        1) After the validation, select the epoch with the best validation score
        2) use the trained model at the selected optimal epoch of validation
        3) perform the evaluate func for the test data
    """
    # start to search for the trained model epoch with the best validation f1 socre
    f1mic_best, ep_best = 0, -1
    
    validation_res_folder = mini_batch_folder + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/validation_trainer_' + str(trainer_id) + '/'
    
    for validation_epoch in snapshot_epoch_list:
        validation_res_file_name = validation_res_folder + 'model_epoch_' + str(validation_epoch)
        with open(validation_res_file_name, "rb") as fp:
            f1mic_val, accuracy_val = pickle.load(fp)
            
        if f1mic_val > f1mic_best:
            f1mic_best, ep_best = f1mic_val, validation_epoch
        
    # use the selected model to perform on the test
    tune_model_folder = mini_batch_folder + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/model_trainer_' + str(trainer_id) + '/'
    
    # save the selected best saved snapshot
    best_model_file = tune_model_folder + 'model_epoch_' + str(ep_best)
    shutil.copy2(best_model_file, tune_model_folder + 'best_saved_snapshot.pkl')
    
    with open(best_model_file, "rb") as fp:
        best_model = pickle.load(fp)
    
    f1, accuracy = Cluster_investigate_evaluation(best_model, mini_batch_folder, eval_target = 'test/test_batch_whole')
    
    # store the resulting data on the disk
    test_res_folder = mini_batch_folder + 'test_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    os.makedirs(os.path.dirname(test_res_folder), exist_ok=True)
    test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
    
    with open(test_res_file, "wb") as fp:
        pickle.dump((f1, accuracy), fp)
    

### ======== Investigate F1-score offline
def execute_investigate_train(mini_batch_folder, tune_param_name, tune_val_label, tune_val, trainer_id = 0, input_layer = [32], epoch_num = 300,
                        dropout = 0.3, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 20, improved = False, diag_lambda = -1,
                              output_period = 10, train_part_num = 2):
    """
        return all validation-F1 for all four models
    """
    # store the resulting data on the disk
    tune_model_folder = mini_batch_folder + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + '/model_trainer_' + str(trainer_id) + '/'
    os.makedirs(os.path.dirname(tune_model_folder), exist_ok=True)
    
    time_res = Cluster_investigate_train(tune_model_folder, mini_batch_folder, input_layer = input_layer, epochs=epoch_num, \
            dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = tune_val, improved = improved, diag_lambda = diag_lambda, \
                              output_period = output_period, train_part_num = train_part_num)
    
    time_res_folder = mini_batch_folder + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/validation_trainer_' + str(trainer_id) + '/'
    os.makedirs(os.path.dirname(time_res_folder), exist_ok=True)
    
    time_res_file_name = time_res_folder + 'time_total_and_load'
    with open(time_res_file_name, "wb") as fp:
        pickle.dump(time_res, fp)
    

def execute_investigate_validation(image_path, mini_batch_folder, tune_param_name, tune_val_label, tune_val, trainer_id = 0, model_epoch = [1]):
    """
        return all validation-F1 for all four models
    """
    # store the resulting data on the disk
    tune_model_folder = mini_batch_folder + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + '/model_trainer_' + str(trainer_id) + '/'
    
    
    res_model_folder = mini_batch_folder + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/validation_trainer_' + str(trainer_id) + '/'
    os.makedirs(os.path.dirname(res_model_folder), exist_ok=True)
    
    for validation_epoch in model_epoch:
        tune_model_file_name = tune_model_folder + 'model_epoch_' + str(validation_epoch)
        with open(tune_model_file_name, "rb") as fp:
            validation_model = pickle.load(fp)
        
        res = Cluster_investigate_evaluation(validation_model, mini_batch_folder)
        # saved the validation results, later can be used in the testing step to choose the trained_model_epoch best validation-F1 score to generate the Test-F1 score
        res_model_file_name = res_model_folder + 'model_epoch_' + str(validation_epoch)
        with open(res_model_file_name, "wb") as fp:
            pickle.dump(res, fp)
      


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


def step2_generate_batch_whole_graph(intermediate_data_path, info_folder = 'info/'): 
    """
        generate all the tensors from the  whole graph for validattion and test
    """
    set_clustering_machine_batch_whole_graph(intermediate_data_path, info_folder = info_folder)


def step30_run_tune_train_batch(intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer,
        trainer_id = 0, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, improved = False, diag_lambda = -1, epoch_num = 400):            
    print('Start running training for partition num: ' + str(train_batch_num))
    # start to tune the model, run different training 
    execute_tuning_train(intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id, input_layer = GCN_layer, epoch_num = epoch_num, \
        dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, improved = improved, diag_lambda = diag_lambda, \
                                  train_batch_num = train_batch_num)
            
def step40_run_tune_test_whole(image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, 
                               train_batch_num, net_layer_num, trainer_id = 0): 
    # set the save path
    print('Start running training for partition num: ' + str(train_batch_num) )
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    img_path += 'tuning_parameters/'  # further subfolder for different task

    # start to validate the model, with different tuning parameters
    execute_tuning_test(img_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)

def step31_run_investigation_train_batch(intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer,
                    trainer_id = 0, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, improved = False, diag_lambda = -1,
                                         epoch_num = 400, output_period = 40):            
    # start to tune the model, and investigate the performance in the middle
    print('Start running investigation traing for train batch num: ' + str(train_batch_num) + ' for_trainer_id_' + str(trainer_id))
    execute_investigate_train(intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id, input_layer = GCN_layer, epoch_num = epoch_num, \
        dropout = dropout, lr = lr, weight_decay = weight_decay, mini_epoch_num = mini_epoch_num, improved = improved, diag_lambda = diag_lambda, \
                              output_period = output_period, train_part_num = train_batch_num)

def step41_run_investigation_validation_batch(image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num,
                    trainer_id = 0, model_epoch = [1]):
    
    print('Start running investigation validation for train batch num: ' + str(train_batch_num) + ' for_trainer_id_' + str(trainer_id))
    img_path = image_data_path + 'cluster_num_' + str(train_batch_num) + '/' + 'net_layer_num_' + str(net_layer_num) + '/'
    
    execute_investigate_validation(img_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id, model_epoch = model_epoch)

    
def step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_path, tune_param_name, tune_val_label_list, tune_val_list, \
                                    train_batch_num, net_layer_num, trainer_list): 
    
    print('Start running training for partition num: ' + str(train_batch_num))
    # set the batch for test and train
    img_path = image_data_path + 'test_res/'  # further subfolder for different task

    # start to summarize the results into images for output

    test_f1, test_accuracy, time_total_train, time_data_load = summarize_tuning_res(img_path, intermediate_data_path, tune_param_name, tune_val_label_list, tune_val_list, trainer_list)

    generate_tuning_raw_data_table(test_accuracy, img_path, 'test_acc.csv', tune_param_name)
    test_accuracy_file = store_data_multi_tuning(tune_val_list, test_accuracy, data_name, img_path, 'accuracy_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(test_accuracy_file, data_name, 'test_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Accuracy')

    generate_tuning_raw_data_table(test_f1, img_path, 'test_f1.csv', tune_param_name)
    test_f1_file = store_data_multi_tuning(tune_val_list, test_f1, data_name, img_path, 'test_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(test_f1_file, data_name, 'vali_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'F1 score')

    generate_tuning_raw_data_table(time_total_train, img_path, 'time_train_total.csv', tune_param_name)
    time_train_file = store_data_multi_tuning(tune_val_list, time_total_train, data_name, img_path, 'train_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(time_train_file, data_name, 'train_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Train Time (ms)')

    generate_tuning_raw_data_table(time_data_load, img_path, 'time_load_data.csv', tune_param_name)
    time_load_file = store_data_multi_tuning(tune_val_list, time_data_load, data_name, img_path, 'load_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num))
    draw_data_multi_tests(time_load_file, data_name, 'load_time_cluster_num_' + str(train_batch_num) + 'net_layer_num_' + str(net_layer_num), 'epochs_per_batch', 'Load Time (ms)')
 
def step51_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_path, tune_param_name, tune_val_label, tune_val, \
                                    train_batch_num, net_layer_num, trainer_list, model_epoch_list): 
    """
        Train investigation post-processing
        Train-validation at the same time
    """
    print('Start summarizing for train batch num: ' + str(train_batch_num) )
    # set the batch for validation and train
    img_path = image_data_path + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    # start to summarize the results into images for output
    
    validation_res_folder = intermediate_data_path + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    Train_peroid_f1, Train_peroid_accuracy = summarize_investigation_distr_res(validation_res_folder, trainer_list, model_epoch_list)

    Train_peroid_f1_file = store_data_each_trainer_investigate(Train_peroid_f1, data_name, 'F1_score', img_path, 'validation')
    draw_data_validation_F1_trainer(Train_peroid_f1_file, data_name, 'validation', 'epoch number', 'F1 score')
    
    Train_peroid_accuracy_file = store_data_each_trainer_investigate(Train_peroid_accuracy, data_name, 'Accuracy', img_path, 'validation')
    draw_data_validation_F1_trainer(Train_peroid_accuracy_file, data_name, 'validation', 'epoch number', 'Accuracy')
    
