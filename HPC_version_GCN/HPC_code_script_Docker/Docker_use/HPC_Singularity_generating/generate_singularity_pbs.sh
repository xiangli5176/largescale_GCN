#PBS -N Sif_container
#PBS -l walltime=00:40:00
#PBS -l nodes=1
#PBS -l mem=16GB
#PBS -j oe
#PBS -A PAS1069

userHome=/users/PAS1069/osu8206


cd $TMPDIR
export SINGULARITY_CACHEDIR=$TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

singularity pull docker://xiangli13257/gpu-pytorch:torch_1_4_geom

mv *.sif ${userHome}/Dockers/sif_container/



cd $PBS_O_WORKDIR
