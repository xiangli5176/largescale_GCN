
# import torch

from multi_exec_code import *


if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    data_name = current_path[-3]
    test_folder_name = 'clusterGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # this is the parts we divide the graph
    origin_train_batch_num = 64
    round_num = 1
    train_batch_num = round_num * origin_train_batch_num

    GCN_layer = [512, 512]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'neuron_num' 

    tune_val_label_list = [32]
    tune_val_list = [ [label] * 2 for label in tune_val_label_list]

    trainer_list = [0]

    # from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit
    # remote_data_root = '~/GCN/Datasets/GraphSaint/'
    # data_class = eval(data_name)
    # dataset = data_class(root = remote_data_root + data_name)
    # data = dataset[0]
    # print_data_info(data, dataset)


    # from torch_geometric.datasets import Planetoid
    # local_data_root = '/media/xiangli/storage1/projects/tmpdata/'
    # dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    # data = dataset[0]




    # investigate train
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step31_run_investigation_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer,
                        trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 10, improved = False, diag_lambda = -1,
                                             epoch_num = 800, output_period = 40)