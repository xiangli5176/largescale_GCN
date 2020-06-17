from multi_exec import *
import dill


if __name__ == "__main__":
    datapath = '/users/PAS1069/osu8206/GCN/Datasets/GraphSaint/'

    working_dir = './res_dir/'
    prepare_data_folder = working_dir + 'prepare_data/'
    img_path = working_dir + 'result/'

    ### ================ Start to do flexible settings according to different dataset: 
    # read the total epoch number from the yml file to determine the mini_epoch_num and eval_train_every
    data_name = 'Flickr'
    train_config_yml = './table2/flickr2_e.yml'


    # =============== Step1 *** prepare for the batches, models, model_evaluation
    train_params, train_phases, train_data, arch_gcn = train_setting(data_name, datapath, train_config_yml)
    prepare(working_dir, train_data, train_params, arch_gcn)
    train_phase_file_name = prepare_data_folder + 'model_train_phase'
    with open(train_phase_file_name, "wb") as fp:
        dill.dump(train_phases, fp)
