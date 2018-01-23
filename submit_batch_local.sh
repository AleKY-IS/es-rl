#!/bin/bash

declare -a inputs=("--max-generations 1"
				   "--max-generations 2"
				   "--max-generations 3")

for i in "${inputs[@]}"
do
   bash run_arg_local.sh $i
done
