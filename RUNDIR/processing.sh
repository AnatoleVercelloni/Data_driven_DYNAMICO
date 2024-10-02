#!/bin/bash

N=6 #number of job

for ((i=0;i<$N;i++))
do
	echo "submitting job for rank $i"
	sbatch --error="prepro_"$i".err" --output="prepro_"$i".out" --export=ALL,rank=$i prepro_job
	#sbatch prepro_job
done
