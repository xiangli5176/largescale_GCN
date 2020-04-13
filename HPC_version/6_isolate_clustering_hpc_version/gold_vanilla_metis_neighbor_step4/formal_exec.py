
# import torch

from multi_exec_code import *


if __name__ == '__main__':
    # data_name = 'Reddit'
    # test_folder_name = 'train_10%_full_neigh/'
    # image_data_path = './results/' + data_name + '/' + test_folder_name
    # intermediate_data_folder = './'
    
    # # this is the parts we divide the graph
    # train_batch_num = 4
    # GCN_layer = [32]
    # net_layer_num = len(GCN_layer) + 1
    # # for non-optimization: hop_layer_num == net_layer_num
    # hop_layer_num = net_layer_num - 1     # for this method, the hop_layer_num is the  layer number of neighbors to generate batch from train seed
    # tune_param_name = 'batch_epoch_num'
    # tune_val_list = [400, 200, 100, 50, 20, 10, 5]
    # trainer_list = list(range(7))


    # from torch_geometric.datasets import Reddit
    # local_data_root = '~/GCN/Datasets/'
    # dataset = Reddit(root = local_data_root + '/' + data_name)
    # data = dataset[0]


    # pc version test on Cora
    data_name = 'Cora'
    test_folder_name = 'flat_memory_save_hpc/train_10%_full_neigh/'
    image_data_path = './results/' + data_name + '/' + test_folder_name
    intermediate_data_folder = './metis_trial_1/'

    # this is the parts we divide the graph
    train_batch_num = 4
    GCN_layer = [32]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num        # for this method, the hop_layer_num is the  layer number of neighbors to generate batch from train seed
    tune_param_name = 'batch_epoch_num'
    tune_val_list = [400]
    # tune_val_list = [10, 5]
    trainer_list = list(range(1))

    from torch_geometric.datasets import Planetoid
    local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]

    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, train_batch_num)

    step1_generate_train_batch(intermediate_data_folder, hop_layer_num, train_frac = 1.0, \
                            batch_range = (0, train_batch_num), info_folder = 'info_train_batch/' )

    step2_generate_validation_whole_graph(intermediate_data_folder, info_folder = 'info_validation_whole/')

    for tune_val in tune_val_list:
        for trainer_id in trainer_list:
            step30_run_tune_train_batch(intermediate_data_folder, tune_param_name, tune_val, train_batch_num, GCN_layer, \
                                trainer_id = trainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400)
            
    for tune_val in tune_val_list:
        for trainer_id in trainer_list:
            step40_run_tune_validation_whole(image_data_path, intermediate_data_folder, tune_param_name, tune_val, train_batch_num, net_layer_num, \
                                trainer_id = trainer_id)

    step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_list, \
                                    train_batch_num, net_layer_num, trainer_list)