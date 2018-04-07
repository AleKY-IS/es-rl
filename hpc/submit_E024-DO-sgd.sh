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
FOO=${CORES:="8"}
FOO=${TIME_LIMIT:="72:00"}

# List of input strings to the call
ID="E024-DO-SGD"
COMMON_IN="--id ${ID} --workers ${CORES} --algorithm sNES --optimize-sigma single --env-name MNIST --optimizer SGD --momentum 0.9 --lr 0.05 --cov-lr 0.05 --perturbations 100 --max-generations 1000 --batch-size 1000 --safe-mutation SUM --chkpt-int 1200 --lr-scheduler ExponentialLR --gamma 1"
declare -a INPUTS=(
					"$COMMON_IN --model MNISTNetDropout"
					"$COMMON_IN --model MNISTNet"
					"$COMMON_IN --model MNISTNetNoBN"
				   )
SCRIPT="run_hpc.sh"
REPEATS=30

# Monitorer
TOTAL_TIME="120:00"
MONITORER_INPUTS="-i $ID -t 4000 -c -s"

# Prompt user to verify correctness
echo "The job submissions will look like this:"
echo ""
echo "bsub -q $QUEUE -J "$ID-monitorer" -W $TOTAL_TIME -n 1 -R "span[hosts=1] rusage[mem=6GB]" -o "$ID-monitorer.log" "sh run_monitorer.sh $MONITORER_INPUTS""
for i in "${!INPUTS[@]}"
do
	NAME="$ID-$i-0"
	echo "bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}""
done
echo ""
echo "Does this look correct? (yes/no): "
read ANSWER
if [ "$ANSWER" != "yes" ]
then
	echo "Ended submission script. No jobs submitted"
	exit 0
fi

# Submit monitoring job
bsub -q $QUEUE -J "$ID-monitorer" -W $TOTAL_TIME -n 1 -R "span[hosts=1] rusage[mem=6GB]" -o "$ID-monitorer.log" "sh run_monitorer.sh $MONITORER_INPUTS"

# Submit each submission type, REPEATS times
# Outer loop over REPEATS makes different groups visible from start when monitoring
for ((j=1; j<=REPEATS; ++j))
do
	# For each input string, submit the job using bsub
	for i in "${!INPUTS[@]}"
	do
		echo ""
		NAME="$ID-$i-$j"
		bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log"" "sh $SCRIPT ${INPUTS[i]}"
	done
done
echo ""

