#PBS -N PyGCN_Reddit
#PBS -l walltime=1:00:00
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

cp -avr ${inputdir}/GCN_tuning ./GCN_tuning
cp -avr ${inputdir}/test ./test


cp ${workdir}/module/*.py .
cp $PBS_O_WORKDIR/*.py .

singularity exec --nv $HOME/Dockers/sif_container/gpu-pytorch_torch_1_4_geom.sif python formal_run_test_id_0.py

cp -avr ./results ${outputdir}/
cp -avr ./info*/ ${outputdir}/


cd $PBS_O_WORKDIR
