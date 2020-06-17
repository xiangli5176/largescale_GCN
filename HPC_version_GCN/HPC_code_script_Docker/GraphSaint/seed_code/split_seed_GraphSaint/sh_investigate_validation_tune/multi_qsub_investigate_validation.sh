#!/bin/bash

# tune_val_list=$(seq 1 4)
trainer_id_list=$(seq 0 2)
epoch_list=$(seq 5 5 15)


for tune_val in 1 5
do
	for trainer_id in ${trainer_id_list}
	do
		target_folder=./tune_val_${tune_val}/trainer_id_${trainer_id}
		cd ${target_folder}
		for epoch_idx in ${epoch_list}
		do
			sub_file=step3_investigate_validation_epoch_idx_${epoch_idx}.sh
			qsub ${sub_file}
			echo "submitted job for the tune value : [${tune_val}]; trainer id : [${trainer_id}] ; epoch_idx : [${epoch_idx}] "
		done
		cd ../..
	done
	
done

echo "all submitted"