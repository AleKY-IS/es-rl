#!/bin/bash

# Exit if error
set -e

# Get inputs
INPUTS=$@

# Run install script
bash hpc_python_setup.sh

# Load modules
module load python3/3.6.2
module load gcc/7.2.0
module load opencv/3.3.1-python-3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2

# Activate ml envrionment
source ~/mlenv/bin/activate

# Set $HOME if running as a bsub script
if [ -z "$BSUB_O_WORKDIR" ]; then
    export HOME=$BSUB_O_WORKDIR
fi
# Set $HOME if running as a qsub script
if [ -z "$PBS_O_WORKDIR" ]; then
    export HOME=$PBS_O_WORKDIR
fi
cd $HOME

# Execute script
python3 ../data-analysis/monitor.py $INPUTS
