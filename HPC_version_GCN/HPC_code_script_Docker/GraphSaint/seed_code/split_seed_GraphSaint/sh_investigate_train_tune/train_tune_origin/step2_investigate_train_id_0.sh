#PBS -N PyGCN_GraphSaint
#PBS -l walltime=00:10:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=96GB
#PBS -j oe
#PBS -A PAS1069


userHome=/users/PAS1069/osu8206
Graph_data_dir=${userHome}/GCN/Datasets/GraphSaint
base_dir=${PBS_O_WORKDIR}/../..

working_dir=$TMPDIR/graphsaint_workdir   # when conducting experiments: tmp dir workdir 
local_dir=${base_dir}/graphsaint_resdir   # local HPC workdir

cp -avr ${base_dir}/module ${working_dir}

cd ${working_dir}

mkdir ./res_dir/
cp -avr ${local_dir}/prepare_data ./res_dir/

cp $PBS_O_WORKDIR/*.py .

singularity exec --nv $HOME/Dockers/sif_container/gpu-pytorch_torch_1_4_geom.sif python formal_investigate_train_id_0.py

cp -avr ./res_dir/model_snapshot ${local_dir}/
cp -avr ./res_dir/result ${local_dir}/

cd $PBS_O_WORKDIR
