# !/bin/bash
# BSUB -J deep-evo-1
# BSUB -q hpc
# BSUB -W 10:30
# BSUB -n 24
# BSUB -R "span[hosts=1]"
# BSUB -R "rusage[mem=6GB]"
# BSUB -o deep-evo-1_%J.log

# Stop on error
set -e

# # Set $HOME if running as a qsub script
# if [ -z "$PBS_O_WORKDIR" ]; then
#     export HOME=$PBS_O_WORKDIR
# fi

# change 
export PATH=/appl/cmake/2.8.12.2/bin:$PATH
cmake --version

# load modules
module load python3/3.6.2
module load gcc/7.2.0
module load opencv/3.3.1-python-3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2

which python3

# # Use HOME directory as base
# cd $HOME

# Setup virtual env
if [ ! -d ~/mlenv ]
then
    python3 -m venv ~/mlenv --copies
    pip3 install matplotlib scikit-learn tensorflow keras ipython pandas gym[all] universe
    pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 
    pip3 install torchvision
fi
source ~/mlenv/bin/activate

echo "Install script successfully completed"
#!/bin/sh
#PBS -N test
#PBS -q hpc
#PBS -l walltime=01:30:00
#PBS -l nodes=1:ppn=20
#PBS -l vmem=6gb
#PBS -j oe
#PBS -o test.log 


