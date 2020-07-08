import shutil
import os

def generate_validation_pbsfile(dst_file, src_file, dst_file_code_name):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.startswith("singularity"):
                    tmpline = line.split()
                    tmpline[-1] = dst_file_code_name
                    line = ' '.join(tmpline) + '\n'
                writeout.write(line)

def generate_src_file(dst_file, src_file, tune_val, trainer_id, epoch_idx):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.strip().startswith("tune_val_label_list"):
                    line = " " * 4 + "tune_val_label_list = [" + str(tune_val) + "]\n"
                elif line.strip().startswith("trainer_list"):
                    line = " " * 4 + "trainer_list = [" + str(trainer_id) + "]\n"
                elif line.strip().startswith("res_model_epoch_list"):
                    line = " " * 4 + "res_model_epoch_list = [" + str(epoch_idx) + "]\n"
                
                writeout.write(line)

if __name__ == '__main__':
    # data path for the train tasks
    datapath = "./"

    tune_val_label_list = [1, 2, 3, 4]
    trainer_id_list = list(range(2))

    epoch_num_list = list(range(10, 201, 10))

    # source path to copy from
    src_data_folder = datapath + "validation_tune_origin/trainer_origin/"
    src_file_py = src_data_folder + "formal_investigate_validation_epoch_idx_0.py"
    src_file_pbs = src_data_folder + "step6_investigate_validation_epoch_idx_0.sh"

    for tune_val_label in tune_val_label_list:
        tune_val_folder = datapath + "tune_val_" + str(tune_val_label) + '/'
        os.makedirs(os.path.dirname(tune_val_folder), exist_ok=True)
        for trainer_id in trainer_id_list:
            trainer_id_folder = tune_val_folder + "trainer_id_" + str(trainer_id) + '/'
            os.makedirs(os.path.dirname(trainer_id_folder), exist_ok=True)
            for epoch_idx in epoch_num_list:
                # copy the python src file
                dst_file_code_name = "formal_investigate_validation_epoch_idx_" + str(epoch_idx) + ".py"
                dst_file_py = trainer_id_folder + dst_file_code_name
                generate_src_file(dst_file_py, src_file_py, tune_val_label, trainer_id, epoch_idx)
                # copy the submission pbs file
                dst_file_pbs = trainer_id_folder + "step6_investigate_validation_epoch_idx_" + str(epoch_idx) + ".sh"
                generate_validation_pbsfile(dst_file_pbs, src_file_pbs, dst_file_code_name)