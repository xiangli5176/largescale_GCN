#!/bin/bash

job_id=9954692
num=5
n=0

while [ $n -lt ${num} ]
do
    qdel ${job_id}
    job_id=$(( job_id+1 ))
    n=$(( n+1 ))
done

 
