#!/bin/bash

inputs=$@
echo "Inputs: $inputs"

source activate ml
python supervised-experiments/mnist_main.py $inputs