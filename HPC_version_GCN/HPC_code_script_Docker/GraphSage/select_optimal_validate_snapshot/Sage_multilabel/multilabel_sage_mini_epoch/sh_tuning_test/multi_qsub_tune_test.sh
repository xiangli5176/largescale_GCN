#!/bin/bash

trainer_id_list=$(seq 0 2)
for tune_val in 1 5 10
# for tune_val in 5
do
	tune_val_folder=./tune_val_${tune_val}
	cd ${tune_val_folder}

	for trainer_id in ${trainer_id_list}
	do
		sub_file=step7_test_run_id_${trainer_id}.sh
		qsub ${sub_file}
		echo "submitted job for the tune value : [${tune_val}]; trainer id : [${trainer_id}]"

	done
	cd ..
done

echo "Test all submitted"