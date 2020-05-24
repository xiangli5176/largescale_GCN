
from multi_exec_code import *


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


    # pc version test on Cora
    data_name = 'Cora'
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
    res_model_epoch_list = list(range(10, 601, 10))

    from torch_geometric.datasets import Planetoid
    local_data_root = '/home/xiangli/projects/tmpdata/GCN/small/'
    dataset = Planetoid(root = local_data_root + 'Planetoid/Cora', name=data_name)
    data = dataset[0]


    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, origin_train_batch_num, validation_ratio = 0.12, test_ratio = 0.22, mini_cluster_num = 32, round_num = 2)

    step1_generate_train_batch(intermediate_data_folder, \
                            batch_range = (0, train_batch_num), 
                            info_folder = 'info_train_batch/' )

    step2_generate_batch_whole_graph(intermediate_data_folder, info_folder = 'info_whole/')


    # tuning the mini-epoch number
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step30_run_tune_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer, \
                trainer_id = trainer_id, dropout = 0.1, lr = 0.001, weight_decay = 0.1, mini_epoch_num = 10, improved = False, diag_lambda = 0.2, epoch_num = 600)
                
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step40_run_tune_test_whole(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                                trainer_id = trainer_id)

                
    step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label_list, tune_val_list, \
                                    train_batch_num, net_layer_num, trainer_list)


    # investigate train
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step31_run_investigation_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer,
                        trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 10, improved = False, diag_lambda = -1,
                                             epoch_num = 600, output_period = 10)


    # investigate validation
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step41_run_investigation_validation_batch(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                        trainer_id = trainer_id, model_epoch = res_model_epoch_list)


    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        step51_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, \
                                    train_batch_num, net_layer_num, trainer_list, res_model_epoch_list)