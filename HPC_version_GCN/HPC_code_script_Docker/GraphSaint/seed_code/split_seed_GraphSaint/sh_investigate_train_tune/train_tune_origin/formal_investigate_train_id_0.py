from multi_exec import *
import dill


if __name__ == "__main__":
    datapath = '/users/PAS1069/osu8206/GCN/Datasets/GraphSaint/'

    working_dir = './res_dir/'
    prepare_data_folder = working_dir + 'prepare_data/'
    img_path = working_dir + 'result/'

    core_par_sampler = 1
    samples_per_processor = -(-200 // core_par_sampler) # round up division
    eval_train_every = 5  # period to record the train loss

    ### ================ Start to do flexible settings according to different dataset: 
    # read the total epoch number from the yml file to determine the mini_epoch_num and eval_train_every
    multilabel_tag = False

    tune_param_name = 'mini_epoch_num'
    tune_val_label_list = [1] 
    tune_val_list = [val for val in tune_val_label_list]

    snapshot_period = 5   # period when to take a snapshot of the model for validation later


    trainer_list = [0]



    # ============== Step2 *** conduct the training process
    train_input_file_name = prepare_data_folder + 'model_train_input'
    with open(train_input_file_name, "rb") as fp:
        minibatch, model = dill.load(fp)


    train_phase_file_name = prepare_data_folder + 'model_train_phase'
    with open(train_phase_file_name, "rb") as fp:
        train_phases = dill.load(fp)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_train_investigate(img_path, working_dir, train_phases, model, minibatch, eval_train_every, 
                                      tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id,
                                      snapshot_every = snapshot_period, mini_epoch_num = 5, multilabel = multilabel_tag, 
                                      core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)
