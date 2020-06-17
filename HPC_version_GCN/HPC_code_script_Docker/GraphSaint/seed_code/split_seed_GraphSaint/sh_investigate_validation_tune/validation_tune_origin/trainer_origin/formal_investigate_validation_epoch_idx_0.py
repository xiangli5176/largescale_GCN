from multi_exec import *
import dill


if __name__ == "__main__":
    datapath = '/users/PAS1069/osu8206/GCN/Datasets/GraphSaint/'

    working_dir = './res_dir/'
    prepare_data_folder = working_dir + 'prepare_data/'
    img_path = working_dir + 'result/'

    ### ================ Start to do flexible settings according to different dataset: 
    # read the total epoch number from the yml file to determine the mini_epoch_num and eval_train_every

    tune_param_name = 'mini_epoch_num'
    tune_val_label_list = [1] 
    tune_val_list = [val for val in tune_val_label_list]

    snapshot_period = 5   # period when to take a snapshot of the model for validation later

    # model_epoch_list = list(range(snapshot_period, 16, snapshot_period))    # snapshot epoch list for validation

    model_epoch_list = [5]
    trainer_list = [0]



    # ================ Step3*** investigate validation:
    evaluation_input_file_name = prepare_data_folder + 'model_eval_input'
    with open(evaluation_input_file_name, "rb") as fp:
        minibatch_eval, model_eval = dill.load(fp)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_validation_investigate(img_path, working_dir, minibatch_eval, model_eval, model_epoch_list, 
                                    tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)

