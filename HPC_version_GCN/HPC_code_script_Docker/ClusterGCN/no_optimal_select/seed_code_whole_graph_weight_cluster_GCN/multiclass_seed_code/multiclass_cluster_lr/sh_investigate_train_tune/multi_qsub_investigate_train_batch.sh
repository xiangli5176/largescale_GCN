#!/bin/bash


for tune_val in 2 3 4
# for tune_val in 4
do
	tune_val_folder=./tune_val_${tune_val}
	cd ${tune_val_folder}
	for trainer_id in 0 1
	do
		sub_file=step5_investigate_train_id_${trainer_id}.sh
		qsub ${sub_file}
		echo "submitted job for the tune value : [${tune_val}] trainer id : [${trainer_id}] "
	done
	cd ..
done

echo "all submitted"