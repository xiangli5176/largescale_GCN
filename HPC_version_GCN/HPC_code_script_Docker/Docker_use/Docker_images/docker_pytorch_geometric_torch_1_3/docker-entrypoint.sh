#!/bin/bash

set -e

export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

if [ "$#" -eq 0 ]
    then
	exec bash
    else 
        exec "$@"
fi

