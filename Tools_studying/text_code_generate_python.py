import shutil
import os

def generate_train_pbsfile(dst_file, src_file, trainer_id):
    update_python_src = "formal_exec_id_" + str(trainer_id) + ".py"
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.startswith("singularity"):
                    tmpline = line.split()
                    tmpline[-1] = update_python_src
                    line = ' '.join(tmpline) + '\n'
                writeout.write(line)

def generate_src_file(dst_file, src_file, tune_val, trainer_id):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.strip().startswith("tune_val_list"):
                    line = " " * 4 + "tune_val_list = [" + str(tune_val) + "]\n"
                elif line.strip().startswith("trainer_list"):
                    line = " " * 4 + "trainer_list = [" + str(trainer_id) + "]\n"
                
                writeout.write(line)

if __name__ == '__main__':
    datapath = "./data_use/text_job/"
    tune_val_list = [10, 5]
    trainer_ids = list(range(1, 7))
    
    src_file_py = datapath + "formal_exec_id_0.py"
    src_file_pbs = datapath + "step3_train_run_id_0.batch"

    for tune_val in tune_val_list:
        tune_val_folder = datapath + "tune_val_" + str(tune_val) + '/'
        os.makedirs(os.path.dirname(tune_val_folder), exist_ok=True)
        for trainer_id in trainer_ids:
            dst_file_py = tune_val_folder + "formal_exec_id_" + str(trainer_id) + ".py"
            generate_src_file(dst_file_py, src_file_py, tune_val, trainer_id)
            dst_file_pbs = tune_val_folder + "step3_train_run_id_" + str(trainer_id) + ".batch"
            generate_train_pbsfile(dst_file_pbs, src_file_pbs, trainer_id)