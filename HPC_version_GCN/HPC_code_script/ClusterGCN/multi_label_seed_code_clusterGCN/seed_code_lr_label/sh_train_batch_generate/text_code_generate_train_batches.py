import shutil
import os

def generate_train_pbsfile(dst_file, src_file, dst_file_code):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.startswith("singularity"):
                    tmpline = line.split()
                    tmpline[-1] =  dst_file_code
                    line = ' '.join(tmpline) + '\n'
                writeout.write(line)

def generate_src_file(dst_file, src_file, batch_group_size, batch_group_id, trainer_id):
    with open(src_file, "r") as readin:
        with open(dst_file, "w") as writeout:
            for line in readin:
                if line.strip().startswith("batch_range"):
                    changed_tuple = (batch_group_id * batch_group_size, (batch_group_id + 1) * batch_group_size)
                    line = " " * 4 * 7 + "batch_range = " + str(changed_tuple) + ", \\" + "\n"
                # elif line.strip().startswith("trainer_list"):
                #     line = " " * 4 + "trainer_list = [" + str(trainer_id) + "]\n"
                
                writeout.write(line)

if __name__ == '__main__':
    # data path for the train tasks
    datapath = "./"
    total_batch = 16
    batch_group_size = 8
    group_num = total_batch // batch_group_size
    batch_group_ids = list(range(1,  group_num))
    trainer_ids = [0]  # for each 0 id, actually it executes for multi id
    # source path to copy from
    src_data_folder = datapath + "batch_group_0/"
    src_file_py = src_data_folder + "formal_train_batches_0.py"
    src_file_pbs = src_data_folder + "step1_generate_train_batch_0.batch"

    for batch_group_id in batch_group_ids:
        batch_group_folder = datapath + "batch_group_" + str(batch_group_id) + '/'
        os.makedirs(os.path.dirname(batch_group_folder), exist_ok=True)
        for trainer_id in trainer_ids:
            dst_file_code_name = "formal_train_batches_" + str(trainer_id) + ".py"
            dst_file_py = batch_group_folder + dst_file_code_name
            generate_src_file(dst_file_py, src_file_py, batch_group_size, batch_group_id, trainer_id )
            dst_file_pbs = batch_group_folder + "step1_generate_train_batch_" + str(trainer_id) + ".batch"
            generate_train_pbsfile(dst_file_pbs, src_file_pbs, dst_file_code_name)