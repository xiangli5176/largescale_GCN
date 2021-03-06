
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


    # pc version test on small version of PPI
    data_name = 'PPI_small'
    test_folder_name = 'GraphSage/'
    intermediate_data_folder = './tuning_lr_template/'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name

    # this is the parts we divide the graph
    train_batch_num = 16

    GCN_layer = [64, 64]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'learning_rate'
    tune_val_label_list = [2, 3, 4]
    tune_val_list = [10**(-label) for label in tune_val_label_list]

    trainer_list = list(range(2))


    from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit, PPI_small
    remote_data_root = '/home/xiangli/projects/tmpdata/GCN/GraphSaint/'
    class_data = eval(data_name)
    dataset = class_data(root = remote_data_root + data_name)
    print('number of data', len(dataset))
    data = dataset[0]
    print_data_info(data, dataset)

    # use the same batch number for both the train and validation processes
    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, train_batch_num = train_batch_num, validation_batch_num = train_batch_num, \
                                  validation_ratio = 0.05, test_ratio = 0.85)

    step1_generate_train_batch(intermediate_data_folder, sample_num = 10, \
                            batch_range = (0, train_batch_num), \
                            info_folder = 'info_train_batch/' )

    step2_generate_test_whole_graph(intermediate_data_folder, info_folder = 'info_test_whole/')

    # tuning the mini-epoch number
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step30_run_tune_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, GCN_layer, \
                                trainer_id = trainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400)
                
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step40_run_tune_test_whole(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, \
                                trainer_id = trainer_id)
                
    step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label_list, tune_val_list, \
                                    train_batch_num, net_layer_num, trainer_list)


    # train-batch investigate
    step31_generate_validation_batch(intermediate_data_folder, sample_num = 10, \
                            batch_range = (0, train_batch_num), 
                            info_folder = 'info_validation_batch/' )

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step32_run_investigation_train_batch(image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer, \
                            trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.1, mini_epoch_num = 20, epoch_num = 400, output_period = 40)


    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        step41_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val, \
                                        train_batch_num, net_layer_num, trainer_list)