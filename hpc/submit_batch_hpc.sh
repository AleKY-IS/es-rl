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
	    -i|--identification)
		    TEST_ID="$2"
		    shift # past argument
		    shift # past value
		    ;;
	    -q|--queue)
		    QUEUE="$2"
		    shift # past argument
		    shift # past value
		    ;;
	    -c|--cores)
		    CORES="$2"
		    shift # past argument
		    shift # past value
		    ;;
	    -t|--timelimit)
		    TIME_LIMIT="$2"
		    shift # past argument
		    shift # past value
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
FOO=${TIME_LIMIT:="10:00"}

for i in {0..$REPEATS..2}
  do 
     echo "Welcome $i times"
 done

# For each input string, submit the job using bsub
for i in "${!INPUTS[@]}"
do
	for ((j=1; j<=REPEATS; ++j))
	do
		echo ""
		NAME="$TEST_ID-$i-$j"
		bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}"
		echo "It had inputs: ${INPUTS[i]}"
		echo "The bash call was:"
		echo "$ bsub -q $QUEUE -J $NAME -W $TIME_LIMIT -n $CORES -R "span[hosts=1] rusage[mem=6GB]" -o "$NAME.log" "sh $SCRIPT ${INPUTS[i]}""

		# bash run_arg_hpc.sh ${INPUTS[i]}
		# bsub < run_arg_hpc.sh
	done
done
echo ""


sleep 5
watch qstat $QUEUE