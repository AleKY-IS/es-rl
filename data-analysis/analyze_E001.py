import argparse
import os

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import torch
from context import utils
from utils.misc import get_equal_dicts, length_of_longest
import utils.plotting as plot
from data_analysis import invert_signs

"""Script for analyzing experiment E005

This experiment examines the scaling of the parallel implementation by running 
different numbers of perturbations on different numbers of CPUs.

From the submission script:

    PERTURBATIONS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
    CORES=(1 2 4 8 12 16 20 24)

Second run of jobs starts at job id 792529
"""

parser = argparse.ArgumentParser(description='Monitorer')
parser.add_argument('-d', type=str, default=None, metavar='--directory', help='The directory of checkpoints to monitor.')
args = parser.parse_args()

if not args.d:
    # args.d = "/home/jakob/mnt/Documents/es-rl/experiments/checkpoints/E001"
    args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/E001-SM"
    # args.d = "/Users/Jakob/mnt/Documents/es-rl/experiments/checkpoints/E001"

save_dir = os.path.join(args.d, 'analysis')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

directories = [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']
filename = 'state-dict-algorithm.pkl'
rm_filenames = ['state-dict-best-algorithm.pkl', 'state-dict-best-optimizer.pkl', 'state-dict-best-model.pkl']

algorithm_states = []
workers = []
perturbations = []
for i, d in enumerate(directories):
    try:
        s = torch.load(os.path.join(d, filename))
        algorithm_states.append(s) 
        print("Loaded " + str(i) + "/" + str(len(directories)) + ": " + d)
    except:
        print("No files found in: " + d)
invert_signs(algorithm_states, keys='all')
    
groups = [0]*len(algorithm_states)
for i, s in enumerate(algorithm_states):
    if s['safe_mutation'] == 'SUM':
        groups[i] = 1
groups = np.array(groups)

for k in ['return_unp', 'return_avg', 'return_min', 'return_max']:

    list_of_series = [s['stats'][k] for s in algorithm_states]
    list_of_genera = [s['stats']['generations'] for s in algorithm_states]
    l = length_of_longest(list_of_series)
    indices = [i for i, series in enumerate(list_of_series) if len(series) == l]

    list_of_series = [list_of_series[i] for i in indices]
    list_of_genera = [list_of_genera[i] for i in indices]
    groups_longest_series = groups[indices]

    plot.timeseries_median_grouped(list_of_genera, list_of_series, groups_longest_series, xlabel='generations', ylabel=k)
    plt.savefig(os.path.join(save_dir, 'analysis-01-' + k + '.pdf'), bbox_inches='tight')
    plt.close()

IPython.embed()