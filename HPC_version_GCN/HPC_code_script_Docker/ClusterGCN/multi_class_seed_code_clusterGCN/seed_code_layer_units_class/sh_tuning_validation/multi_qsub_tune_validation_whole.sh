#!/bin/bash

for tune_val in 1 5 10 20 40 100 200 400
# for tune_val in 5
do
	tune_val_folder=./tune_val_${tune_val}
	cd ${tune_val_folder}

	sub_file=step4_validation_run_id_0.batch
	qsub ${sub_file}
	echo "Validation submitted job for the tune value : [${tune_val}]"

	cd ..
done

echo "Validation all submitted"