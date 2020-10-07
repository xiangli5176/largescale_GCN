import pickle
import shutil
import os

from base_exec import *
from utils import *
from Post_utils import *


# multiexec module:
def execute_train_investigate(image_path, work_dir, train_phases, model, minibatch, eval_train_every, 
                              tune_param_name, tune_val_label, tune_val, trainer_id = 0,
                         snapshot_every = 5, mini_epoch_num = 5, multilabel = True, core_par_sampler = 1, samples_per_processor = 200):
    """
        return all validation-F1 for all four models
    """
    # run the training process for the model
    tune_model_folder = work_dir + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                        '/model_trainer_' + str(trainer_id) + '/'
    
    os.makedirs(os.path.dirname(tune_model_folder), exist_ok=True)
    
    # to apply any tuning values
    total_time_train, time_upload = train_investigate(tune_model_folder, train_phases, model, minibatch, eval_train_every, 
                                    snapshot_every = snapshot_every, mini_epoch_num = tune_val, multilabel = multilabel, 
                              core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
    
    time_info_folder = image_path + 'train_res/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/model_trainer_' + str(trainer_id) + '/'
    
    os.makedirs(os.path.dirname(time_info_folder), exist_ok=True)
    
    time_info_file_name = time_info_folder + 'train_time'
    with open(time_info_file_name, "wb") as fp:
        pickle.dump((total_time_train, time_upload), fp)
            
    
def execute_validation_investigate(image_path, work_dir, minibatch_eval, model_eval, snapshot_epoch_list, 
                                 tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    """
        Perform the validaiton offline from saved snapshot of the models
        snapshot_epoch_list :  a list of the saved models to perform evaluation
    """
    
    tune_model_folder = work_dir + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/model_trainer_' + str(trainer_id) + '/'
    
    validation_res_folder = image_path + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/validation_trainer_' + str(trainer_id) + '/'
    
    os.makedirs(os.path.dirname(validation_res_folder), exist_ok=True)
    
    # start evaluation:
    for validation_epoch in snapshot_epoch_list:
        
        res = evaluate(tune_model_folder, minibatch_eval, model_eval, validation_epoch)
        
        validation_res_file_name = validation_res_folder + 'model_epoch_' + str(validation_epoch)
        with open(validation_res_file_name, "wb") as fp:
            pickle.dump(res, fp)

            
    
def execute_test_tuning(image_path, work_dir, minibatch_eval, model_eval, snapshot_epoch_list, 
                                 tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    """
        1) After the validation, select the epoch with the best validation score
        2) use the trained model at the selected optimal epoch of validation
        3) perform the evaluate func for the test data
    """
    # start to search for the trained model epoch with the best validation f1 socre
    f1mic_best, ep_best = 0, -1
    validation_res_folder = image_path + 'validation_res/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/validation_trainer_' + str(trainer_id) + '/'
    
    for validation_epoch in snapshot_epoch_list:
        validation_res_file_name = validation_res_folder + 'model_epoch_' + str(validation_epoch)
        with open(validation_res_file_name, "rb") as fp:
            f1mic_val, f1mac_val = pickle.load(fp)
        
        if f1mic_val > f1mic_best:
            f1mic_best, ep_best = f1mic_val, validation_epoch
        
    # use the selected model to perform on the test
    tune_model_folder = work_dir + 'model_snapshot/tune_' + tune_param_name + '_' + str(tune_val_label) + \
                                    '/model_trainer_' + str(trainer_id) + '/'
    
    # return 1) micro-f1 ;  2) macro-f1
    res = evaluate(tune_model_folder, minibatch_eval, model_eval, ep_best, mode = 'test')
    
    # save the selected best saved snapshot
    best_model_file = tune_model_folder + 'snapshot_epoch_' + str(ep_best) + '.pkl'
    shutil.copy2(best_model_file, tune_model_folder + 'best_saved_snapshot.pkl')
    
    # store the resulting data on the disk
    test_res_folder = image_path + 'test_res/tune_' + tune_param_name + '_' + str(tune_val_label) + '/'
    os.makedirs(os.path.dirname(test_res_folder), exist_ok=True)
    test_res_file = test_res_folder + 'res_trainer_' + str(trainer_id)
    
    with open(test_res_file, "wb") as fp:
        pickle.dump(res, fp)
    
    
    
