#PBS -N PyGCN_Reddit
#PBS -l walltime=00:40:00
#PBS -l nodes=1:gpus=1
#PBS -l mem=96GB
#PBS -j oe
#PBS -A PAS1069



userHome=/users/PAS1069/osu8206
workshop=${userHome}/GCN/KDD_cluster_GCN_graphsaint_data
workdir=${PBS_O_WORKDIR}/../..

inputdir=${workdir}/input_data
outputdir=${workdir}/output_data

cd $TMPDIR
cp -avr ${inputdir}/data_info ./data_info
cp -avr ${inputdir}/train ./train
# cp -avr ${inputdir}/validation ./validation

cp ${workdir}/module/*.py .
cp $PBS_O_WORKDIR/*.py .

singularity exec --nv $HOME/Dockers/sif_container/gpu-pytorch_torch_1_4_geom.sif python formal_investigate_train_id_0.py


cp -avr ./model_snapshot ${inputdir}/
cp -avr ./info*/ ${outputdir}/

cd $PBS_O_WORKDIR
