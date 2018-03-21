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
FOO=${TIME_LIMIT:="60:00"}



# List of input strings to the call
ID="E005"
SCRIPT="run_hpc.sh"
REPEATS=1
declare -a PERTURBATIONS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
declare -a CORES=(4 8 12 16 20 24)

# Prompt user to verify correctness
echo "The job submissions will look like this:"
echo ""
for j in "${!CORES[@]}"
do
    for i in "${!PERTURBATIONS[@]}"
    do
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --env-name MNIST --model MNISTNet --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
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
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --env-name MNIST --model MNISTNet --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""
declare -a PERTURBATIONS=(2 4 8 16 32 64 128 256 512 1024 2048)
declare -a CORES=(1 2)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""
bsub -q hpc -J "E005-1-16384" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-1-16384.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 10 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 16384 --workers 1"

bsub -q hpc -J "E005-1-8192" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-1-8192.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 20 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 8192 --workers 1"

bsub -q hpc -J "E005-1-4096" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-1-4096.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 40 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 4096 --workers 1"

bsub -q hpc -J "E005-2-16384" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-2-16384.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 20 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 16384 --workers 2"

bsub -q hpc -J "E005-2-8192" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-2-8192.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 40 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 8192 --workers 2"

bsub -q hpc -J "E005-2-8192" -W "60:00" -n 16 -R "span[hosts=1] rusage[mem=6GB]" -o "E005-2-8192.log" "sh run_hpc.sh --id E005 --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 100 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations 8192 --workers 2"


declare -a PERTURBATIONS=(32768)
declare -a CORES=(2 4 8 12 16 20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

declare -a PERTURBATIONS=(65536)
declare -a CORES=(4 8 12 16 20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

declare -a PERTURBATIONS=(131072)
declare -a CORES=(8 12 16 20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

declare -a PERTURBATIONS=(262144)
declare -a CORES=(12 16 20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

declare -a PERTURBATIONS=(524288)
declare -a CORES=(16 20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""

declare -a PERTURBATIONS=(1048576)
declare -a CORES=(20 24)
for j in "${!CORES[@]}"
do
	# For each input string, submit the job using bsub
	for i in "${!PERTURBATIONS[@]}"
	do
		echo ""
        NAME="$ID-${CORES[j]}-${PERTURBATIONS[i]}"
        INPUT="--id ${ID} --algorithm ES --optimizer SGD --lr-scheduler ExponentialLR --gamma 0.99970 --env-name MNIST --max-generations 64 --batch-size 200 --safe-mutation SUM --chkpt-int 10000 --perturbations ${PERTURBATIONS[i]} --workers ${CORES[j]}"
	    bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT $INPUT"
		echo "Submission : $ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n ${CORES[j]} -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "
		echo "Script call: $SCRIPT $INPUT"
	done
done
echo ""
