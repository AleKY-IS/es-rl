#!/bin/bash

# List of input strings to the call
declare -a INPUTS=(
				   "--max-generations 5000 --batch-size 1000 --lr-scheduler ExponentialLR --gamma 0.9995"
				   "--max-generations 5000 --batch-size 1000 --lr-scheduler ExponentialLR --gamma 0.9995 --safe-mutation"
				   )
SCRIPT="run_hpc.sh"
REPEATS=10

# Parse inputs
POSITIONAL=()
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
	    -n|--name)
		    TEST_ID="$2"
		    shift # past argument
		    shift # past value
		    ;;
	    -q|--queue)
		    QUEUE="$2"
		    shift
		    shift
		    ;;
	    -c|--cores)
		    CORES="$2"
		    shift
		    shift
		    ;;
	    -t|--timelimit)
		    TIME_LIMIT="$2"
		    shift
		    shift
		    ;;
		-h|--help)
			echo "Help for submit_batch_hpc.sh script"
			echo "This script submits a batch of jobs to the HPC cluster."
			echo "Options:"
			echo "	-h, --help       Display help"
			echo "	-n, --name       Name of the job"
			echo "	-q, --queue      The HPC queue to submit to"
			echo "	-c, --cores      The number of cores to use for execution"
			echo "	-t, --timelimit  The wall clock time limit of the job after which it is terminated"
			exit # end script if help displayed
			;;
	    *)    # unknown option
		    POSITIONAL+=($key) # save it in an array for later
		    shift # past argument
		    shift # past value
		    ;;
	esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo $POSITIONAL

# Set defaults if unassigned
FOO=${TEST_ID:="default-name"}
FOO=${QUEUE:="hpc"}
FOO=${CORES:="20"}
FOO=${TIME_LIMIT:="15:00"}

# For each input string, submit the job using bsub
for i in "${!INPUTS[@]}"
do
	for ((j=1; j<=REPEATS; ++j))
	do
		echo ""
		NAME="$TEST_ID-$i-$j"
		bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}"
		# source activate ml
		# python ../experiments/main.py ${INPUTS[i]}
		echo "It had inputs: ${INPUTS[i]}"
		echo "The bash call was:"
		echo "$ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}""
	done
done
echo ""
