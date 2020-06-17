#PBS -N Py_GraphSaint
#PBS -l walltime=00:10:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=96GB
#PBS -j oe
#PBS -A PAS1069



userHome=/users/PAS1069/osu8206
Graph_data_dir=${userHome}/GCN/Datasets/GraphSaint
base_dir=${PBS_O_WORKDIR}/..

working_dir=$TMPDIR/graphsaint_workdir   # when conducting experiments: tmp dir workdir 
local_dir=${base_dir}/graphsaint_resdir   # local HPC workdir

cp -avr ${base_dir}/module ${working_dir}

cd ${working_dir}


cp $PBS_O_WORKDIR/*.py .

singularity exec --nv $HOME/Dockers/sif_container/gpu-pytorch_torch_1_4_geom.sif python step1_prepare_model.py

cp -avr ./res_dir/ ${local_dir}/

cd $PBS_O_WORKDIR
