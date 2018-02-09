#!/bin/bash

# Exit if error
set -e

# Parse inputs
POSITIONAL=()
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
	    -i|--id)
		    ID="$2"
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
FOO=${ID:="E999-default"}
FOO=${QUEUE:="hpc"}
FOO=${CORES:="20"}
FOO=${TIME_LIMIT:="24:00"}



# List of input strings to the call
SCRIPT="run_hpc.sh"
REPEATS=1
declare -a PERTUBATIONS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
declare -a CORES=(1 2 4 8 12 16 20 24)

# Prompt user to verify correctness
echo "The job submissions will look like this:"
echo ""
for j in "${!CORES[@]}"
do
    for i in "${!PERTUBATIONS[@]}"
    do
        ID="E004-scaling-n${CORES[j]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 1000 --safe-mutation SUM --chkpt-int 10000 --pertubations ${PERTUBATIONS[i]}"
	    echo "bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT""
    done
done
echo ""
echo "Does this look correct? (yes/no): "
read ANSWER
if [ "$ANSWER" != "yes" ]
then
	echo "Ended submission script. No jobs submitted"
	exit 0
fi

# Submit each submission type, REPEATS times
# Outer loop over REPEATS makes different groups visible from start when monitoring
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTUBATIONS[@]}"
	do
		echo ""
        ID="E004-scaling-n${CORES[j]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 1000 --safe-mutation SUM --chkpt-int 10000 --pertubations ${PERTUBATIONS[i]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

