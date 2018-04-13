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

# Append correct cmake version to path
export PATH=/appl/cmake/2.8.12.2/bin:$PATH
cmake --version

# load modules
module load python3/3.6.2
module load gcc/7.2.0
module load opencv/3.3.1-python-3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20
module load scipy/0.19.1-python-3.6.2

# # Use HOME directory as base
# cd $HOME

# Set open file limit
ulimit -Sn 32768

# Setup virtual env
if [ ! -d ~/ml ]
then
    python3 -m venv ~/ml --copies
    source ~/ml/bin/activate
    pip3 install -U matplotlib scikit-learn tensorflow keras ipython pandas seaborn dropbox
    pip3 install -U requests>=2.18.1  # Version to fix "float() argument must be a string or a number, not 'Timeout'" error in _validate_timeout in urllib3/utils.timeout.py l. 124
    pip3 install -U gym==0.9.5  # Version to fix bug with missing gym.benchmarks
    pip3 install -U gym[box2d]
    pip3 install -U gym[atari]
    pip3 install -U universe
    pip3 install -U http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl  # http://download.pytorch.org/whl/cu90/torch-0.3.1.post4-cp36-cp36m-linux_x86_64.whl
    pip3 install -U torchvision
fi
source ~/ml/bin/activate

echo "Install script successfully completed"
