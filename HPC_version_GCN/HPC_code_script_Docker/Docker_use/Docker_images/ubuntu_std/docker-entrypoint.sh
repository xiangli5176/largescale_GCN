#!/bin/bash

# set -e

export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# if no command input: then run the bash of the container
# otherwise run the command as provided by the :
# --nv: to enable gpu
# for a pytorch usually run by python : the inputfile can come from your local 
# storage on OSC
# singularity run --nv <image name> python file 
if [ "$#" -eq 0 ]
    then
	exec bash
    else 
        exec "$@"
fi

