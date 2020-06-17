import shutil
import os

def generate_train_pbsfile(dst_file, src_file, dst_file_code_name):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.startswith("singularity"):
                    tmpline = line.split()
                    tmpline[-1] = dst_file_code_name
                    line = ' '.join(tmpline) + '\n'
                writeout.write(line)

def generate_src_file(dst_file, src_file, tune_val, trainer_id):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.strip().startswith("tune_val_label_list"):
                    line = " " * 4 + "tune_val_label_list = [" + str(tune_val) + "]\n"
                elif line.strip().startswith("trainer_list"):
                    line = " " * 4 + "trainer_list = [" + str(trainer_id) + "]\n"
                
                writeout.write(line)

if __name__ == '__main__':
    # data path for the train tasks
    datapath = "./"
    # tune_val_label_list = [4]
    # trainer_ids = list(range(1, 2))

    tune_val_label_list = [2, 3, 4]
    trainer_ids = list(range(2))

    # source path to copy from
    src_data_folder = datapath + "tune_train_origin/"
    src_file_py = src_data_folder + "formal_investigate_train_id_0.py"
    src_file_pbs = src_data_folder + "step5_investigate_train_id_0.sh"

    for tune_val_label in tune_val_label_list:
        tune_val_folder = datapath + "tune_val_" + str(tune_val_label) + '/'
        os.makedirs(os.path.dirname(tune_val_folder), exist_ok=True)
        for trainer_id in trainer_ids:
            dst_file_code_name = "formal_investigate_train_id_" + str(trainer_id) + ".py"
            dst_file_py = tune_val_folder + dst_file_code_name
            generate_src_file(dst_file_py, src_file_py, tune_val_label, trainer_id)
            dst_file_pbs = tune_val_folder + "step5_investigate_train_id_" + str(trainer_id) + ".sh"
            generate_train_pbsfile(dst_file_pbs, src_file_pbs, dst_file_code_name)