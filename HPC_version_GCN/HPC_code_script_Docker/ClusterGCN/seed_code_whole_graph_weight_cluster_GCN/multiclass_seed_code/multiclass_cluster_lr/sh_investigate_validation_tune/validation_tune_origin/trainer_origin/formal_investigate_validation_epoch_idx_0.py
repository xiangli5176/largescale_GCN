
# import torch

from multi_exec_code import *


if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    # this -4 corresponds to the directory depth
    data_name = current_path[-4]
    test_folder_name = 'hack_clusterGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    
    # this is the parts we divide the graph
    origin_train_batch_num = 32
    round_num = 1
    train_batch_num = round_num * origin_train_batch_num

    GCN_layer = [256, 256]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'learning_rate' 

    tune_val_label_list = [4]
    tune_val_list = [10**(-label) for label in tune_val_label_list]

    trainer_list = [0]

    res_model_epoch_list = [0]
    # res_model_epoch_list = list(range(40, 801, 40))

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



                        
    # investigate validation
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step41_run_investigation_validation_batch(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                        trainer_id = trainer_id, model_epoch = res_model_epoch_list)