#PBS -N PyGCN_GraphSaint
#PBS -l walltime=00:10:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=96GB
#PBS -j oe
#PBS -A PAS1069


userHome=/users/PAS1069/osu8206
Graph_data_dir=${userHome}/GCN/Datasets/GraphSaint
base_dir=${PBS_O_WORKDIR}/../..    # locate the directory

working_dir=$TMPDIR/graphsaint_workdir   # when conducting experiments: tmp dir workdir 
local_dir=${base_dir}/graphsaint_resdir   # local HPC workdir

cp -avr ${base_dir}/module ${working_dir}

cd ${working_dir}

mkdir -p ./res_dir/prepare_data
cp ${local_dir}/prepare_data/model_eval_input ./res_dir/prepare_data
mkdir -p ./res_dir/result
cp -avr ${local_dir}/result/validation_res ./res_dir/result

cp -avr ${local_dir}/model_snapshot ./res_dir/

cp $PBS_O_WORKDIR/*.py .

singularity exec --nv $HOME/Dockers/sif_container/gpu-pytorch_torch_1_4_geom.sif python formal_run_test_id_0.py

cp -avr ./res_dir/result/test_res ${local_dir}/result

cd $PBS_O_WORKDIR
