import argparse
import os

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import torch
from context import utils
from utils.misc import get_equal_dicts


"""Script for analyzing experiment E005

This experiment examines the scaling of the parallel implementation by running 
different numbers of perturbations on different numbers of CPUs.

From the submission script:

    PERTURBATIONS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
    CORES=(1 2 4 8 12 16 20 24)

Second run of jobs starts at job id 792529
"""

ONLY_LOGARITHMICALLY_SPACED_WORKER_TIMES = False

parser = argparse.ArgumentParser(description='Monitorer')
parser.add_argument('-d', type=str, default=None, metavar='--directory', help='The directory of checkpoints to monitor.')
args = parser.parse_args()

if not args.d:
    # args.d = "/home/jakob/mnt/Documents/es-rl/experiments/checkpoints/E005"
    # args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/E005-sca"
    args.d = "/Users/Jakob/mnt/Documents/es-rl/experiments/checkpoints/E005"

directories = [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']
filename = 'state-dict-algorithm.pkl'
rm_filenames = ['state-dict-best-algorithm.pkl', 'state-dict-best-optimizer.pkl', 'state-dict-best-model.pkl']

algorithm_states = []
workers = []
perturbations = []
for d in directories:
    try:
        s = torch.load(os.path.join(d, filename))
        algorithm_states.append(s)
        workers.append(s['workers'])
        perturbations.append(s['perturbations'])
        print(s['workers'], s['perturbations'])
    except:
        print("None in: " + d)
    # for rmf in rm_filenames:
    #     try:
    #         IPython.embed()
    #         os.remove(os.path.join(d, rmf))
    #     except OSError:
    #         pass
workers = np.array(workers)
perturbations = np.array(perturbations)

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

# Sort according to number of workers. This makes lines in the plots nice
sort_indices = np.argsort(workers)
workers = workers[sort_indices]
perturbations = perturbations[sort_indices]
mean_of_times = mean_of_times[sort_indices]
std_of_times = std_of_times[sort_indices]
mean_of_parallel_fractions = mean_of_parallel_fractions[sort_indices]
std_of_parallel_fractions = std_of_parallel_fractions[sort_indices]

if ONLY_LOGARITHMICALLY_SPACED_WORKER_TIMES:
    keep_vals = [1,2,4,8,16,24]
    keep_ids = []
    for v in keep_vals:
        keep_ids.extend(list(np.where(workers == v)[0]))
    keep_ids = np.array(keep_ids)
    workers = workers[keep_ids]
    perturbations = perturbations[keep_ids]
    mean_of_times = mean_of_times[keep_ids]
    std_of_times = std_of_times[keep_ids]
    mean_of_parallel_fractions = mean_of_parallel_fractions[keep_ids]
    std_of_parallel_fractions = std_of_parallel_fractions[keep_ids]

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


# Amdahl's law test
# Compute speed up for certain choice of number of perturbations. The higher the better parallel fraction.
# However, use only perturbations completed with a single worker since T1 only exists for these
perturbations_SP = perturbations[workers == 1]
TP = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
TP_std = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
SP = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
SP_std = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
workers_SP = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
f = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
f_std = {k: np.zeros(len(np.unique(workers))) for k in np.unique(perturbations)}
for i_pert in sorted(np.unique(perturbations_SP), reverse=False):
    # Indices of this perturbation
    ids_perturbation = perturbations == i_pert
    # The worker indices
    workers_SP[i_pert] = workers[ids_perturbation]
    SP[i_pert] = np.zeros(len(workers_SP[i_pert]))
    SP_std[i_pert] = np.zeros(len(workers_SP[i_pert]))
    f[i_pert] = {k: np.zeros(len(workers_SP[i_pert])) for k in np.unique(perturbations)}
    # Indices of the minimal number of workers that did this perturbation
    ids_minimal_workers_with_i_pert = workers == 1  # workers == workers_SP[i_pert].min()
    # Index of this job
    ids_minimal_workers_i_pert = np.logical_and(ids_minimal_workers_with_i_pert, ids_perturbation)
    if not ids_minimal_workers_i_pert.any():
        continue
    # Time per generation and of this job
    T1 = mean_of_times[ids_minimal_workers_i_pert]
    T1_std = std_of_times[ids_minimal_workers_i_pert]
    # Compute speed-ups
    TP[i_pert] = mean_of_times[ids_perturbation]
    TP_std[i_pert] = std_of_times[ids_perturbation]
    # Speed-up
    SP[i_pert] = T1/TP[i_pert]
    SP_std[i_pert] = np.sqrt((T1_std / T1)**2 + (TP_std[i_pert] / TP[i_pert])**2) * np.abs(SP[i_pert])  # See Taylor: "Error analysis"
    # Parallel fraction
    f[i_pert] = (1 - TP[i_pert]/T1) / (1 - 1/workers_SP[i_pert])
    f[i_pert][0] = 0
    f_std[i_pert] = np.zeros(f[i_pert].shape) # TODO: Compute
    



# Plotting
# Figure 1: Average time per generation vs number of workers, logarithmic y
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = mean_of_times[ids]
    s = std_of_times[ids]
    legend.append(str(i_pert))
    ax.errorbar(x, y, yerr=s, fmt='-o')
plt.xlabel('Number of workers')
plt.ylabel('Average time per generation [s]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(legend, title='Perturbations', loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_yscale('log')
plt.savefig(os.path.join(args.d,'E005-scaling-01.pdf'), bbox_inches='tight')

# Figure 2: Average time per generation vs number of workers, logarithmic y and x
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = mean_of_times[ids]
    s = std_of_times[ids]
    legend.append(str(i_pert))
    ax.errorbar(x, y, yerr=s, fmt='-o')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(legend, title='Perturbations', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Number of workers')
plt.ylabel('Average time per generation [s]')
ax.set_yscale('log')
ax.set_xscale('log', basex=2)
plt.savefig(os.path.join(args.d,'E005-scaling-02.pdf'), bbox_inches='tight')

# Figure 3: Average time per generation vs number of workers, logarithmic y and x, smoothed interpolation
fig, ax = plt.subplots()
ax.set_prop_cycle(None)
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = mean_of_times[ids]
    s = std_of_times[ids]
    try:
        tck = sp.interpolate.splrep(x, y, k=3)
    except TypeError:
        continue
    x_interpolated = np.linspace(x.min(),x.max(),300)
    y_smoothed = sp.interpolate.splev(x_interpolated, tck, der=0)
    legend.append(str(i_pert) + ' perturbations')
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(x_interpolated, y_smoothed, color=color)
    ax.errorbar(x, y, yerr=s, fmt='o', color=color)
plt.xlabel('Number of workers')
plt.ylabel('Time per generation [s]')
# ax.legend(legend)
ax.set_yscale('log')
ax.set_xscale('log', basex=2)
plt.savefig(os.path.join(args.d,'E005-scaling-03.pdf'), bbox_inches='tight')

# Figure 4: Amdahl's law test
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    if (workers_SP[i_pert] == 0).all():
        continue
    legend.append(str(i_pert))
    ax.errorbar(workers_SP[i_pert], SP[i_pert], yerr=SP_std[i_pert], fmt='-o')
ax.errorbar(workers_SP[2], workers_SP[2], fmt='-')
legend.append("Ideal")
plt.xlabel('Number of workers')
plt.ylabel('Speed-up factor')
ax.legend(legend, title='Perturbations', loc='upper left', ncol=2, columnspacing=0.5)
plt.savefig(os.path.join(args.d,'E005-scaling-04.pdf'), bbox_inches='tight')

# Figure 5: Amdahl's law test
fig, ax = plt.subplots()
legend = []
for i_workers in sorted(np.unique(workers)):
    SP_max = []
    SP_max_std = []
    perturbations_max_SP = []
    for i_pert in sorted(perturbations_SP):
        idx = workers_SP[i_pert] == i_workers
        if idx.any():
            SP_max.append(SP[i_pert][idx])
            SP_max_std.append(SP_std[i_pert][idx])
            perturbations_max_SP.append(i_pert)
    SP_max = np.array(SP_max)
    SP_max_std = np.array(SP_max_std)
    perturbations_max_SP = np.array(perturbations_max_SP)
    ax.errorbar(perturbations_max_SP, SP_max, yerr=SP_max_std, fmt='-o')
    legend.append(str(i_workers))
ax.set_xscale('log', basex=2)
plt.xlabel('Number of perturbations')
plt.ylabel('Speed-up factor')
plt.legend(legend, title='CPUs')
plt.savefig(os.path.join(args.d,'E005-scaling-05.pdf'), bbox_inches='tight')

# Figure 6: ...
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = mean_of_parallel_fractions[ids]
    s = std_of_parallel_fractions[ids]
    legend.append(str(i_pert) + ' perturbations')
    ax.errorbar(x, y, yerr=s, fmt='-o')
plt.xlabel('Number of workers')
plt.ylabel('Average parallel fraction')
ax.legend(legend)
# ax.set_yscale('log')
plt.savefig(os.path.join(args.d,'E005-scaling-06.pdf'), bbox_inches='tight')

# Figure 7: Amdahl's law test: Parallel fraction as function of perturbations
fig, ax = plt.subplots()
legend = []
for i_workers in sorted(np.unique(workers)):
    x = np.array([])
    y = np.array([])
    y_std = np.array([])
    for i_pert in perturbations_SP:
        idx = workers_SP[i_pert] == i_workers
        y = np.append(y, f[i_pert][idx])
        y[y<0] = 0
        y_std = np.append(y_std, f_std[i_pert][idx])
    ax.errorbar(perturbations_SP, y, yerr=y_std, fmt='o')
    legend.append(str(i_workers))
plt.xlabel('Number of perturbations')
plt.ylabel('Average parallel fraction')
ax.legend(legend, title='CPUs')
ax.set_xscale('log', basex=2)
# ax.set_yscale('log', basey=10)
plt.savefig(os.path.join(args.d,'E005-scaling-07.pdf'), bbox_inches='tight')



fig, ax = plt.subplots()
ax.plot(range(0,10000), range(10000, 0, -1))
ax.set_yscale('log')
# plt.savefig(os.path.join(args.d,'E005-scaling-03.pdf'), bbox_inches='tight')

