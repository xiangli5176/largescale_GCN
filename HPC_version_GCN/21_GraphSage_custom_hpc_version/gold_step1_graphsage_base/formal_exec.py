from multi_exec import *
import dill
from Post_utils import *


if __name__ == "__main__":
    datapath = '/home/xiangli/projects/tmpdata/GCN/GraphSaint/'

    working_dir = './res_lr_0001/'
    prepare_data_folder = working_dir + 'prepare_data/'
    img_path = working_dir + 'result/'

    core_par_sampler = 1
    samples_per_processor = -(-200 // core_par_sampler) # round up division
    eval_train_every = 5  # period to record the train loss

    ### ================ Start to do flexible settings according to different dataset: 
    # read the total epoch number from the yml file to determine the mini_epoch_num and eval_train_every
    data_name = 'Flickr'
    train_config_yml = './table2/flickr2_sage.yml'
    multilabel_tag = False

    # data_name = 'PPI_small'
    # train_config_yml = './table2/ppi2_e.yml'


    tune_param_name = 'mini_epoch_num'
    tune_val_label_list = [1, 5] 
    tune_val_list = [val for val in tune_val_label_list]

    snapshot_period = 2   # period when to take a snapshot of the model for validation later

    # refer to the yml file to decide the training period:
    model_epoch_list = list(range(snapshot_period, 16, snapshot_period))    # snapshot epoch list for validation

    trainer_list = list(range(3))



    # =============== Step1 *** prepare for the batches, models, model_evaluation
    train_params, train_phases, train_data, arch_gcn = train_setting(data_name, datapath, train_config_yml)
    prepare(working_dir, train_data, train_params, arch_gcn)
    train_phase_file_name = prepare_data_folder + 'model_train_phase'
    with open(train_phase_file_name, "wb") as fp:
        dill.dump(train_phases, fp)

    
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
                                      snapshot_every = snapshot_period, mini_epoch_num = 5, multilabel = multilabel_tag, input_neigh_deg = [10, 5],
                                      core_par_sampler = core_par_sampler, samples_per_processor = samples_per_processor)

    # ================ Step3*** investigate validation:
    evaluation_input_file_name = prepare_data_folder + 'model_eval_input'
    with open(evaluation_input_file_name, "rb") as fp:
        minibatch_eval, model_eval = dill.load(fp)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_validation_investigate(img_path, working_dir, minibatch_eval, model_eval, model_epoch_list, 
                                    tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)


    # ================= Step4*** investigate test:
    evaluation_input_file_name = prepare_data_folder + 'model_eval_input'
    with open(evaluation_input_file_name, "rb") as fp:
        minibatch_eval, model_eval = dill.load(fp)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            execute_test_tuning(img_path, working_dir, minibatch_eval, model_eval, model_epoch_list, 
                                    tune_param_name, tune_val_label, tune_val, trainer_id = trainer_id)


    # =================Step5 Post processing ===============
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step51_run_investigation_summarize_whole(data_name, img_path,
                                         tune_param_name, tune_val_label, tune_val,
                                            trainer_list, model_epoch_list)

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_list:
            step50_run_tune_summarize_whole(data_name, img_path, 
                                    tune_param_name, tune_val_label_list, tune_val_list,
                                    trainer_list)
