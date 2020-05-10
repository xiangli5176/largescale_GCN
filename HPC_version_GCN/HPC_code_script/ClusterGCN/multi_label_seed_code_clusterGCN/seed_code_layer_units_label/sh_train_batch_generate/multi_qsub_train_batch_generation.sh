#!/bin/bash

for batch_group in 0 1
do
	batch_group_folder=./batch_group_${batch_group}
	cd ${batch_group_folder}

	sub_file=step1_generate_train_batch_0.batch
	qsub ${sub_file}
	echo "Train batches generation submitted job for the batch group : [${batch_group}]"

	cd ..
done

echo "Train batches generation all submitted"