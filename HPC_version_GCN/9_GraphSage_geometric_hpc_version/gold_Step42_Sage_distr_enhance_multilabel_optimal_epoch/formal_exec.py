
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
    data_name = 'PPI_small'
    test_folder_name = 'Graph_Sage/'
    intermediate_data_folder = './tuning_lr_template/'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name

    # this is the parts we divide the graph
    train_batch_num = 8
    validation_batch_num = 2
    test_batch_num = 2

    GCN_layer = [256]
    net_layer_num = len(GCN_layer) + 1
    # for non-optimization: hop_layer_num == net_layer_num
    hop_layer_num = net_layer_num

    tune_param_name = 'learning_rate'
    tune_val_label_list = [3, 4]
    tune_val_list = [10**(-label) for label in tune_val_label_list]

    trainer_list = list(range(2))
    res_model_epoch_list = list(range(10, 201, 10))
    validation_batch_ids = list(range(validation_batch_num))
    test_batch_ids = list(range(test_batch_num))


    from GraphSaint_dataset import print_data_info, Flickr, Yelp, PPI_large, Amazon, Reddit, PPI_small
    remote_data_root = '/home/xiangli/projects/tmpdata/GCN/GraphSaint/'
    class_data = eval(data_name)
    dataset = class_data(root = remote_data_root + data_name)
    print('number of data', len(dataset))
    data = dataset[0]
    print_data_info(data, dataset)

    step0_generate_clustering_machine(data, dataset, intermediate_data_folder, train_batch_num = train_batch_num, validation_batch_num = train_batch_num,
                                  validation_ratio = 0.12, test_ratio = 0.22)

    step1_generate_train_batch(intermediate_data_folder, sample_num = 10,
                            batch_range = (0, train_batch_num), 
                            info_folder = 'info_train_batch/' )


    set_clustering_machine_eval_batch(intermediate_data_folder, sample_num = 10, layer_num = hop_layer_num, eval_type = "validation",
                                      batch_range = (0, validation_batch_num), info_folder = 'info_eval_batch/')

    set_clustering_machine_eval_batch(intermediate_data_folder, sample_num = 10, layer_num = hop_layer_num, eval_type = "test",
                                      batch_range = (0, test_batch_num), info_folder = 'info_eval_batch/')

    
    ### ====================== investigate train
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for tainer_id in trainer_list:
            step31_run_investigation_train_batch(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, train_batch_num, net_layer_num, GCN_layer,
                                                trainer_id = tainer_id, dropout = 0.1, lr = 0.0001, weight_decay = 0.01, mini_epoch_num = 10, improved = True, diag_lambda = -1,
                                                epoch_num = 200, output_period = 10)

    # investigate validation, scatter the tasks among validation batches
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_investigate_validation_scatter_distr(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, 
                                                    trainer_id = trainer_id, model_epoch = res_model_epoch_list, batch_ids = validation_batch_ids)
            
    # investigate validation, aggregate results from distributed validation results
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_investigate_validation_aggr_distr(intermediate_data_folder, tune_param_name, tune_val_label, tune_val,
                                                trainer_id = trainer_id, model_epoch = res_model_epoch_list, batch_ids = validation_batch_ids)
    
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        step51_run_investigation_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label, tune_val,
                                        train_batch_num, net_layer_num, trainer_list, res_model_epoch_list)


    ### ================== investigate test, scatter the tasks among test batches
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_test_tuning_scatter_distr(intermediate_data_folder, res_model_epoch_list, tune_param_name, tune_val_label, tune_val, 
                                        trainer_id = trainer_id, batch_ids = test_batch_ids)
            
    # investigate test, aggregate results from distributed test results
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_test_tuning_aggr_distr(intermediate_data_folder, tune_param_name, tune_val_label, tune_val, 
                                    trainer_id = trainer_id, batch_ids = test_batch_ids)

    step50_run_tune_summarize_whole(data_name, image_data_path, intermediate_data_folder, tune_param_name, tune_val_label_list, tune_val_list, \
                                train_batch_num, net_layer_num, trainer_list)                          