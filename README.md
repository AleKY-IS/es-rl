# es-rl
Training of neural networks using variations of 'evolutionary' methods including the 'Evolutionary Strategy' presented by OpenAI and Variational Optimization.

## Local installation
To create a new environment with the required packages, run 

```conda env create -f environment.yml```

or to update an existing environment to include the required packages, run

```conda env update --file environment.yml```

Any of these two commands will create an Anaconda virtual environment called `ml`


## HPC installation
To run the code on the High Performance Computing Cluster at the Technical University of Denmark first of all requires a user login.

### Pip
The easiest way to create the environment on the HPC is using pip.

The script `hpc_python_setup.sh` will setup up the environment. The environment is called `mlenv` in this case.

### Anaconda
Anaconda can be installed on the HPC. Get the latest 64 bit x86 version from <https://www.anaconda.com/download/#linux>. 
1. Move the downloaded `.sh` file to the root of the HPC.
2. Install Anaconda by calling `bash Anaconda3-5.0.1-Linux-x86_64.sh` at the root.
3. Follow the installation instructions.

My personal root directory is `/zhome/c2/b/86488/`

## Executing jobs on HPC
### Connecting
A connection to the HPC can be established by SSH by 
```ssh s132315@login3.hpc.dtu.dk```

A local mirror of the user folder on the HPC can be created by `sshfs` 
```sshfs s132315@login.hpc.dtu.dk:/zhome/c2/b/86488 ~/mnt -o defer_permissions -o volname=hpc```

### Submitting
A single job can be run (not submitted) by executing the `run_hpc.sh` script. 

An entire batch of jobs can be submitted using the `submit_batch_hpc.sh` script. The specific inputs to each of the jobs must be specified in this script in the `INPUTS` array.
An example call to `submit_batch_hpc.sh` which is

```bash submit_batch_hpc.sh -i SM-experiment -t 10:00 -c 24 -q hpc```

This will submit a series of jobs named "SM-experiment-[id]" with wall clock time limit of 10 hours, requesting a 24 core machine on the hpc queue 

### Monitoring
The `data-analysis/monitor.py` script allows for monitoring of multiple jobs running in parallel, e.g. on the HPC. The script takes a directory of checkpoints as input and uses the saved `stats.pkl` file.

It saves summarizing plots in the source checkpoint folder and displays statistics in the console.
