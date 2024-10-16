#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Illegal number of parameters, parameters should be:  name_processing" >&2
    exit 2
fi

name_processing=$1

if [ $name_processing = "first_dataset" ]
then
	echo "processing first_dataset"
	resolution="low"                # resolution: either low or high
	stride=7                        # stride mean we take 1 file over stride  
	shift_=0                        # shift_ is a shift from the first file 
	first_file=0                    # first file selected
	last_file=183960                # last file selected (210240 files in total)
	#7th year: 183960
	N=2 #number of job

elif [ $name_processing = "scoring_set" ]
then
	echo "processing scoring_set"
        resolution="low"                # resolution: either low or high
        stride=1                        # stride mean we take 1 file over stride  
        shift_=0                        # shift_ is a shift from the first file 
        first_file=183960                    # first file selected
        last_file=210240               # last file selected (210240 files in total)
        #7th year: 183960
        N=2 #number of job

elif [ $name_processing = "scoring_set_hr" ]
then
        echo "processing scoring_set_hr"
        resolution="high"                # resolution: either low or high
        stride=47                        # stride mean we take 1 file over strid>
        shift_=0                        # shift_ is a shift from the first file
        first_file=183960                    # first file selected
        last_file=210240               # last file selected (210240 files in to>
        #7th year: 183960
        N=4 #number of job

else

	echo "processing custom dataset, parameters has to be set !"
	resolution="to_set"
	stride=-1
	shift_=-1
	first_file=-1
	last_file=-1
	N=1
fi



for ((i=0;i<$N;i++))
do
	echo "submitting job for rank $i"
	sbatch --error="prepro_"$i".err" --output="prepro_"$i".out" --export=ALL,name_processing=$name_processing,resolution=$resolution,stride=$stride,shift_=$shift_,first_file=$first_file,last_file=$last_file,rank=$i,N=$N prepro_job
	#sbatch prepro_job
done
