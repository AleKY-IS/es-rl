import argparse
import os

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import torch
from context import utils
from utils.misc import get_equal_dicts


parser = argparse.ArgumentParser(description='Monitorer')
parser.add_argument('-d', type=str, default=None, metavar='--directory', help='The directory of checkpoints to monitor.')
args = parser.parse_args()

if not args.d:
    # args.d = "/home/jakob/mnt/Documents/es-rl/experiments/checkpoints/E006-LVBS"
    args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/E005-sca"

directories = [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']
filename = 'state-dict-algorithm.pkl'
rm_filenames = ['state-dict-best-algorithm.pkl', 'state-dict-best-optimizer.pkl', 'state-dict-best-model.pkl']


def set_workers_from_checkpoint_folder(folder, algorithm_state):
    general_form = "MNIST-180211-172630.694268-n"
    l = len(general_form)
    algorithm_state['n_workers'] = int(folder[l:])

algorithm_states = []
n_workers = []
pertubations = []
for d in directories:
    try:
        s = torch.load(os.path.join(d, filename))
        algorithm_states.append(s)
        set_workers_from_checkpoint_folder(d.split(os.sep)[-1], s)
        n_workers.append(s['n_workers'])
        pertubations.append(s['pertubations'])
        print(s['n_workers'], s['pertubations'])
    except:
        print("None in: " + d)
    # for rmf in rm_filenames:
    #     try:
    #         IPython.embed()
    #         os.remove(os.path.join(d, rmf))
    #     except OSError:
    #         pass
n_workers = np.array(n_workers)
pertubations = np.array(pertubations)

# Get data
abs_walltimes = [np.array(s['stats']['walltimes']) + s['stats']['start_time'] for s in algorithm_states]
generation_times = [np.diff(np.array([s['stats']['start_time']] + list(awt))) for s, awt in zip(algorithm_states, abs_walltimes)]
parallel_fractions = [(np.array(s['stats']['workertimes'])/gt) for s, gt in zip(algorithm_states, generation_times)]

# Compute mean and std over groups
mean_of_times = np.array([])
std_of_times = np.array([])
mean_of_parallel_fractions = np.array([])
std_of_parallel_fractions = np.array([])
associated_bs = np.array([])
for i in range(0, len(algorithm_states)):
    mean_of_times = np.append(mean_of_times, np.mean(generation_times[i]))
    std_of_times = np.append(std_of_times, np.std(generation_times[i]))
    mean_of_parallel_fractions = np.append(mean_of_parallel_fractions, np.mean(parallel_fractions[i]))
    std_of_parallel_fractions = np.append(std_of_parallel_fractions, np.std(parallel_fractions[i]))

# Confidence intervals
cis_times = []
for m, s in zip(mean_of_times, std_of_times):
    interval = sp.stats.norm.interval(0.95, loc=m, scale=s)
    half_width = (interval[1] - interval[0])/2
    cis_times.append(half_width)
cis_par_frac = []
for m, s in zip(mean_of_parallel_fractions, std_of_parallel_fractions):
    interval = sp.stats.norm.interval(0.95, loc=m, scale=s)
    half_width = (interval[1] - interval[0])/2
    cis_par_frac.append(half_width)

# Plotting
fig, ax = plt.subplots()
legend = []
for n_pert in sorted(list(np.unique(pertubations)), reverse=True):
    idx = np.where(pertubations == n_pert)
    x = n_workers[idx]
    y = mean_of_times[idx]
    s = std_of_times[idx]
    legend.append(str(n_pert) + ' pertubations')
    ax.errorbar(x, y, yerr=s, fmt='o')
plt.xlabel('Number of workers')
plt.ylabel('Time per generation [s]')
ax.legend(legend)
ax.set_yscale('log')
plt.savefig(os.path.join(args.d,'E005-scaling-01.pdf'))

fig, ax = plt.subplots()
legend = []
for n_pert in [44]:
    idx = np.where(pertubations == n_pert)
    x = n_workers[idx]
    y = mean_of_times[idx]
    s = std_of_times[idx]
    legend.append(str(n_pert) + ' pertubations')
    ax.errorbar(x, y, yerr=s, fmt='o')
plt.xlabel('Number of workers')
plt.ylabel('Time per generation [s]')
ax.legend(legend)
# ax.set_yscale('log')
# ax.set_xscale('log')
plt.savefig(os.path.join(args.d,'E005-scaling-02.pdf'))

fig, ax = plt.subplots()
ax.plot(range(0,10000), range(10000, 0, -1))
ax.set_yscale('log')
plt.savefig(os.path.join(args.d,'E005-scaling-03.pdf'))

IPython.embed()