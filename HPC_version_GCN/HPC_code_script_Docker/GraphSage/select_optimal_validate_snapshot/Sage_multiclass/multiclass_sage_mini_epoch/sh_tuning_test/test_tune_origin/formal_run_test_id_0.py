
# import torch

from multi_exec_code import *


if __name__ == '__main__':
    current_path = os.environ.get('PBS_O_WORKDIR').split('/')
    # this -3 corresponds to the directory depth
    data_name = current_path[-3]
    test_folder_name = 'clusterGCN/'
    intermediate_data_folder = './'
    image_data_path = intermediate_data_folder + 'results/' + data_name + '/' + test_folder_name
    

    tune_param_name = 'mini_epoch_num' 

    tune_val_label_list = [5]
    tune_val_list = [label for label in tune_val_label_list]

    trainer_list = [0]

    res_model_epoch_list = list(range(10, 101, 10))

    # investigate test
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_test_tuning(image_data_path, intermediate_data_folder, res_model_epoch_list, 
                                    tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)
