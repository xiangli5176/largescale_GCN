#!/bin/bash

for tune_val in 1 5 10 20
# for tune_val in 5
do
	tune_val_folder=./tune_val_${tune_val}
	cd ${tune_val_folder}

	sub_file=step4_test_run_id_0.sh
	qsub ${sub_file}
	echo "Test submitted job for the tune value : [${tune_val}]"

	cd ..
done

echo "Test all submitted"