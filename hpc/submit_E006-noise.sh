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



# This experiment is to examine the effect of the batch size on the variance on the obtained loss for a fixed network (learning rate 0) (Loss variance batch size = LVBN).
# In total this experiment requires 28000 generations

# List of input strings to the call
ID="E006-LVBS"
COMMON_IN="--id ${ID} --algorithm ES --optimizer SGD --env-name MNIST --max-generations 100 --lr 0 --pertubations 40 --safe-mutation SUM"
declare -a INPUTS=(
                   "$COMMON_IN --batch-size 10"
                   "$COMMON_IN --batch-size 25"
                   "$COMMON_IN --batch-size 50"
                   "$COMMON_IN --batch-size 100"
                   "$COMMON_IN --batch-size 200"
                   "$COMMON_IN --batch-size 300"
                   "$COMMON_IN --batch-size 400"
                   "$COMMON_IN --batch-size 500"
                   "$COMMON_IN --batch-size 600"
                   "$COMMON_IN --batch-size 700"
                   "$COMMON_IN --batch-size 800"
                   "$COMMON_IN --batch-size 900"
                   "$COMMON_IN --batch-size 1000"
				   "$COMMON_IN --batch-size 1"
                   "$COMMON_IN --batch-size 2"
                   "$COMMON_IN --batch-size 4"
                   "$COMMON_IN --batch-size 8"
                   "$COMMON_IN --batch-size 16"
                   "$COMMON_IN --batch-size 32"
                   "$COMMON_IN --batch-size 64"
                   "$COMMON_IN --batch-size 128"
                   "$COMMON_IN --batch-size 256"
                   "$COMMON_IN --batch-size 512"
                   "$COMMON_IN --batch-size 1024"
                   "$COMMON_IN --batch-size 2048"
                   "$COMMON_IN --batch-size 4192"
                   "$COMMON_IN --batch-size 8384"
                   "$COMMON_IN --batch-size 16768"
				   )
SCRIPT="run_hpc.sh"
REPEATS=10


# Monitorer
#let TOTAL_TIME=24*${#INPUTS[@]}*$REPEATS/8
#TOTAL_TIME="$TOTAL_TIME:00"
TOTAL_TIME=432:00
MONITORER_INPUTS="-d $ID -t 120 -c"

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
# bsub -q hpc -J E004-ant-monitorer -W 90:00 -n 1 -R "span[hosts=1] rusage[mem=6GB]" -o E004-ant-monitorer.log "sh run_monitorer.sh -d E004-ant -t 120 -c"
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
		# source activate ml
		# python ../experiments/main.py ${INPUTS[i]}
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log"" # "sh $SCRIPT ${INPUTS[i]}""
		echo "Script call: $SCRIPT ${INPUTS[i]}"
	done
done
echo ""

