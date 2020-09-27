import pickle
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv




def generate_tuning_raw_data_table(data_dict, file_path, file_name, tune_param_name):
    """
        data_dict : a dictionary of different runing index with different tuning values
                data_dict[1]: e.g.  index 1 runing, this is a dictionary of tuning values
    """
    target_file = file_path + file_name
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = [tune_param_name] + list(data_dict[0].keys())
        wr.writerow(header)
        for i, tune_val in data_dict.items():
            tmp_line = [i] + [tune_val[key] for key in tune_val.keys()]
            wr.writerow(tmp_line)


# post processing to generate the data figure for tracing the validation f1-score
def summarize_investigation_distr_res(image_path, trainer_list, model_epoch_list):
    """
        return:
        Train_peroid_f1(a dict of dict) : for each trainer, a dict of f1 for each epoch timepoint during training
    """
    Train_peroid_f1 = {}
    Train_peroid_accuracy = {}
    
    for trainer_id in trainer_list:
        validation_res_folder = image_path + 'validation_trainer_' + str(trainer_id) + '/'
        
        f1_epoch = {}
        accuracy_epoch = {}
        for validation_epoch in model_epoch_list:
            validation_res_file_name = validation_res_folder + 'model_epoch_' + str(validation_epoch)
            with open(validation_res_file_name, "rb") as fp:
                f1_epoch[validation_epoch], accuracy_epoch[validation_epoch] = pickle.load(fp)
        
        Train_peroid_f1[trainer_id], Train_peroid_accuracy[trainer_id] = f1_epoch, accuracy_epoch
        
    return Train_peroid_f1, Train_peroid_accuracy     


def store_data_each_trainer_investigate(investigate_res, data_name, res_name, img_path, comments):
    """
        investigate_res: currently either F1-score or accuracy a dict {epoch num : value}
    """
    
    pickle_filename = img_path + data_name + '_' + res_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    
    with open(pickle_filename, "wb") as fp:
        pickle.dump(investigate_res, fp)
    return pickle_filename


def draw_data_validation_F1_trainer(pickle_filename, data_name, comments, xlabel, ylabel):
    """
        Draw the figure for from the stored data (multiple store functions)
    """
    with open(pickle_filename, "rb") as fp:
        res_trainer = pickle.load(fp)
    
    for trainer_id, F1_track in res_trainer.items():
        img_name = pickle_filename[:-4] + '_img_trainer_' + str(trainer_id)
        
        plt.clf()
        plt.figure()
        sns.set(style='whitegrid')
        
        Validation_F1 = {}
        Validation_F1[xlabel] = sorted(F1_track.keys())
        Validation_F1[ylabel] = [F1_track[key] for key in Validation_F1[xlabel]]
        df = pd.DataFrame(Validation_F1) 
        
        g = sns.relplot(x = xlabel, y = ylabel, markers=True, kind="line", data=df)
        g.despine(left=True)
        g.fig.suptitle(data_name + ' ' + ylabel + ' ' + comments)
        g.set_xlabels(xlabel)
        g.set_ylabels(ylabel)

        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        plt.savefig(img_name, bbox_inches='tight')


def step51_run_investigation_summarize_whole(data_name, image_data_path,
                                         tune_param_name, tune_val_label, tune_val,
                                            trainer_list, model_epoch_list): 
    """
        Train investigation post-processing
    """
    print('Start summarizing for dataset : ' + str(data_name) )
    img_path = image_data_path + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'

    validation_micro_f1, validation_macro_f1 = summarize_investigation_distr_res(img_path, trainer_list, model_epoch_list)

    validation_micro_f1_file = store_data_each_trainer_investigate(validation_micro_f1, data_name, 'micro_F1_score', img_path, 'validation')
    draw_data_validation_F1_trainer(validation_micro_f1_file, data_name, 'validation', 'epoch number', 'micro F1 score')
    
    validation_macro_f1_file = store_data_each_trainer_investigate(validation_macro_f1, data_name, 'macro_F1_score', img_path, 'validation')
    draw_data_validation_F1_trainer(validation_macro_f1_file, data_name, 'validation', 'epoch number', 'macro F1 score')




### ================================================================================

# post-processing for tuning on test results
def summarize_tuning_res(image_path, trainer_list, 
                         tune_param_name, tune_val_label_list, tune_val_list):
    """
        tune_val_label_list :  label of the tuning parameter value for file location
        tune_val_list  :       the real value of the tuning parameter
    """
    test_micro_f1 = {}
    test_macro_f1 = {}
    time_total_train = {}
    time_data_load = {}
    
    res = []
    for trainer_id in trainer_list:
        ref = {}
        for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
            test_res_folder = image_path + 'test_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
            test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
            with open(test_res_file, "rb") as fp:
                ref[tune_val] = list(pickle.load(fp))
            
            # append the total time of train, and the time of data loading in sequence
            train_time_folder = image_path + 'train_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
            train_time_file = train_time_folder  + 'model_trainer_' + str(trainer_id) + '/train_time'
            with open(train_time_file, "rb") as fp:
                ref[tune_val].extend( list(pickle.load(fp)) )
            
        res.append(ref)
    
    for i, ref in enumerate(res):
        test_micro_f1[i] = {tune_val : res_lst[0] for tune_val, res_lst in ref.items()}
        test_macro_f1[i] = {tune_val : res_lst[1] for tune_val, res_lst in ref.items()}
        time_total_train[i] = {tune_val : res_lst[2] for tune_val, res_lst in ref.items()}
        time_data_load[i] = {tune_val : res_lst[3] for tune_val, res_lst in ref.items()}
        
    return test_micro_f1, test_macro_f1, time_total_train, time_data_load


def store_data_multi_tuning(tune_params, target, data_name, img_path, comments):
    """
        tune_params: is the tuning parameter list
        target: is the result, here should be F1-score, accuraycy, load time, train time
    """
    run_ids = sorted(target.keys())   # key is the run_id
    run_data = {'run_id': run_ids}
    # the key can be converted to string or not: i.e. str(tune_val)
    # here we keep it as integer such that we want it to follow order
    tmp = {tune_val : [target[run_id][tune_val] for run_id in run_ids] for tune_val in tune_params}  # the value is list
    run_data.update(tmp)
    
    pickle_filename = img_path + data_name + '_' + comments + '.pkl'
    os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
    df = pd.DataFrame(data=run_data, dtype=np.int32)
    df.to_pickle(pickle_filename)
    return pickle_filename


def draw_data_multi_tests(pickle_filename, data_name, comments, xlabel, ylabel):
    df = pd.read_pickle(pickle_filename)
    df_reshape = df.melt('run_id', var_name = 'model', value_name = ylabel)

    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')
    g = sns.catplot(x="model", y=ylabel, kind='box', data=df_reshape)
    g.despine(left=True)
    g.fig.suptitle(data_name + ' ' + ylabel + ' ' + comments)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    img_name = pickle_filename[:-4] + '_img'
    
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    plt.savefig(img_name, bbox_inches='tight')
    

    
def step50_run_tune_summarize_whole(data_name, image_data_path, 
                                    tune_param_name, tune_val_label_list, tune_val_list,
                                    trainer_list): 
    
    print('Start running training for dataset: ' + str(data_name))
    # set the batch for test and train
    tuning_res_path = image_data_path + 'test_res/'  # further subfolder for different task

    # start to summarize the results into images for output

    test_micro_f1, test_macro_f1, time_total_train, time_data_load = summarize_tuning_res(image_data_path, trainer_list, 
                                                                            tune_param_name, tune_val_label_list, tune_val_list)

    generate_tuning_raw_data_table(test_macro_f1, tuning_res_path, 'test_macro_f1.csv', tune_param_name)
    test_macro_f1_file = store_data_multi_tuning(tune_val_list, test_macro_f1, data_name, tuning_res_path, 'macro_f1' )
    draw_data_multi_tests(test_macro_f1_file, data_name, 'test', 'epochs_per_batch', 'macro_f1')

    generate_tuning_raw_data_table(test_micro_f1, tuning_res_path, 'test_micro_f1.csv', tune_param_name)
    test_micro_f1_file = store_data_multi_tuning(tune_val_list, test_micro_f1, data_name, tuning_res_path, 'micro_f1')
    draw_data_multi_tests(test_micro_f1_file, data_name, 'test', 'epochs_per_batch', 'micro_f1')

    generate_tuning_raw_data_table(time_total_train, tuning_res_path, 'time_train_total.csv', tune_param_name)
    time_train_file = store_data_multi_tuning(tune_val_list, time_total_train, data_name, tuning_res_path, 'train_time')
    draw_data_multi_tests(time_train_file, data_name, 'Total_train_time', 'epochs_per_batch', 'Train Time (ms)')

    generate_tuning_raw_data_table(time_data_load, tuning_res_path, 'time_load_data.csv', tune_param_name)
    time_load_file = store_data_multi_tuning(tune_val_list, time_data_load, data_name, tuning_res_path, 'load_time' )
    draw_data_multi_tests(time_load_file, data_name, 'load_time', 'epochs_per_batch', 'Load Time (ms)')