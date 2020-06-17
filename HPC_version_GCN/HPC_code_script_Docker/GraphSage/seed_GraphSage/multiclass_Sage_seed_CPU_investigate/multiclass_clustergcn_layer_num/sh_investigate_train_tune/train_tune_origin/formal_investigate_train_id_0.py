
# import torch

from multi_exec_code import *


if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    data_name = current_path[-3]
    test_folder_name = 'SageGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # this is the parts we divide the graph
    train_batch_num = 64

    GCN_layer = [512, 512]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'layer_num' 

    tune_val_label_list = [4]
    tune_val_list = [ [512] * label for label in tune_val_label_list]

    trainer_list = [0]

    # from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit
    # remote_data_root = '~/GCN/Datasets/GraphSaint/'
    # data_class = eval(data_name)
    # dataset = data_class(root = remote_data_root + data_name)
    # data = dataset[0]
    # print_data_info(data, dataset)

    # # pc version test on Cora
    # data_name = 'Cora'
    # test_folder_name = 'flat_memory_save_hpc/train_10%_full_neigh/'
    # intermediate_data_folder = './trial_KDD/'
    # image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name

    # # this is the parts we divide the graph
    # origin_train_batch_num = 4
    # GCN_layer = [32]
    # net_layer_num = len(GCN_layer) + 1
    # # for non-optimization: hop_layer_num == net_layer_num
    # hop_layer_num = net_layer_num
    # tune_param_name = 'batch_epoch_num'
    # # tune_val_list = [400, 200, 100, 50, 20, 10, 5]
    # tune_val_list = [10, 5]
    # trainer_list = list(range(2))
    # round_num = 2
    # train_batch_num = round_num * origin_train_batch_num

    # from torch_geometric.datasets import Planetoid
    # local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    # dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    # data = dataset[0]


    # step0_generate_clustering_machine(data, dataset, intermediate_data_folder, test_ratio = 0.10, validation_ratio = 0.24, origin_train_batch_num, mini_cluster_num = 1500, round_num = round_num)

    # step1_generate_train_batch(intermediate_data_folder, \
    #                         batch_range = (0, train_batch_num), \
    #                         info_folder = 'info_train_batch/' )

    # step2_generate_validation_whole_graph(intermediate_data_folder, info_folder = 'info_validation_whole/')

    # # tuning the mini-epoch number
    # for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
    #     for trainer_id in trainer_list:
    #         step30_run_tune_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer, \
    #                             trainer_id = trainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400)
            
    # for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
    #     for trainer_id in trainer_list:
    #         step40_run_tune_validation_whole(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
    #                             trainer_id = trainer_id)

    # step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label_list, tune_val_list, \
    #                                 train_batch_num, net_layer_num, trainer_list)


    # investigate train
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step31_run_investigation_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer, \
                            trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 10, 
                            epoch_num = 800, output_period = 40)