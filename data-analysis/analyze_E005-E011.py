import argparse
import os
from distutils.dir_util import copy_tree

import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import torch
from context import utils
from utils.misc import get_equal_dicts
from utils.data_analysis import load_stats, invert_signs


"""Script for analyzing experiment E005 and E011

This experiment examines the scaling of the parallel implementation by running 
different numbers of perturbations on different numbers of CPUs.

From the submission script:

    PERTURBATIONS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
    CORES=(1 2 4 8 12 16 20 24)

Second run of jobs starts at job id 792529
"""

ONLY_LOGARITHMICALLY_SPACED_WORKER_TIMES = False
matplotlib.rcParams.update({'font.size': 12})

parser = argparse.ArgumentParser(description='Monitorer')
parser.add_argument('-d', type=str, default=None, metavar='--directory', help='The directory of checkpoints to monitor.')
args = parser.parse_args()

Eid = 'E011'

if not args.d:
    # args.d = "/home/jakob/mnt/Documents/es-rl/experiments/checkpoints/E005"   # Linux HPC
    # args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/E005-sca"     # Linux
    # args.d = "/home/jakob/mnt/Documents/es-rl/experiments/checkpoints/E011"   # Linux HPC
    # args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/" + Eid + "-sca"     # Linux 

    # dst_dir = '/home/jakob/Dropbox/Apps/ShareLaTeX/Master\'s Thesis/graphics/' + Eid + '-sca-analysis'

    # args.d = "/Users/Jakob/mnt/Documents/es-rl/experiments/checkpoints/" + Eid + "-sca"
    args.d = "/Users/Jakob/Dropbox/es-rl/experiments/checkpoints/" + Eid + "-sca"
    dst_dir = '/Users/Jakob/Dropbox/Apps/ShareLaTeX/Master\'s Thesis/graphics/' + Eid + '-sca-analysis'


save_dir = os.path.join(args.d, Eid + '-sca-analysis')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

directories = [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']
rm_filenames = ['state-dict-best-algorithm.pkl', 'state-dict-best-optimizer.pkl', 'state-dict-best-model.pkl']

algorithm_states = []
stats = []
workers = []
perturbations = []
for d in directories:
    try:
        s = torch.load(os.path.join(d, 'state-dict-algorithm.pkl'))
        if Eid == 'E011' or Eid == 'E005' and s['perturbations'] > 64:
            perturbations.append(s['perturbations'])
            workers.append(s['workers'])
            algorithm_states.append(s)
            stats.append(load_stats(os.path.join(d, 'stats.csv')))
            print(s['workers'], s['perturbations'])
    except:
        print("None in: " + d)
    # for rmf in rm_filenames:
    #     try:
    #         IPython.embed()
    #         os.remove(os.path.join(d, rmf))
    #     except OSError:
    #         pass
invert_signs(stats)
workers = np.array(workers)
perturbations = np.array(perturbations)

for s in stats:
    # Computations/Transformations
    psuedo_start_time = s['walltimes'].diff().mean()
    # Add pseudo start time to all times
    abs_walltimes = s['walltimes'] + psuedo_start_time
    # Append pseudo start time to top of series and compute differences
    s['time_per_iteration'] = pd.concat([pd.Series(psuedo_start_time), abs_walltimes]).diff().dropna()
    s['parallel_fraction'] = s['workertimes']/s['time_per_iteration']

# Compute mean and std over groups
time_per_iteration_means = np.array([])
time_per_iteration_stds = np.array([])
parallel_fraction_means = np.array([])
parallel_fraction_stds = np.array([])
associated_bs = np.array([])
worker_perturbation_pairs = []
for i in range(0, len(algorithm_states)):
    wp_pair = (algorithm_states[i]['workers'], algorithm_states[i]['perturbations'])
    if wp_pair in worker_perturbation_pairs:
        idx = worker_perturbation_pairs.index(wp_pair)
        time_per_iteration_means[idx] = np.min([time_per_iteration_means[idx], np.mean(stats[i]['time_per_iteration'])])
        time_per_iteration_stds[idx] = np.std(time_per_iteration_means[idx])
        parallel_fraction_means[idx] = np.min([parallel_fraction_means[idx], np.mean(stats[i]['parallel_fraction'])])
        parallel_fraction_stds[idx] = np.std(parallel_fraction_means[idx])

        # time_per_iteration_means[idx] = (time_per_iteration_means[idx] + np.mean(stats[i]['time_per_iteration']))/2
        # time_per_iteration_stds[idx] = (time_per_iteration_stds[idx] + np.std(stats[i]['time_per_iteration']))/2
        # parallel_fraction_means[idx] = (parallel_fraction_means[idx] + np.mean(stats[i]['parallel_fraction']))/2
        # parallel_fraction_stds[idx] = (parallel_fraction_stds[idx] + np.std(stats[i]['parallel_fraction']))/2
    else:
        worker_perturbation_pairs.append(wp_pair)
        time_per_iteration_means = np.append(time_per_iteration_means, np.mean(stats[i]['time_per_iteration']))
        time_per_iteration_stds = np.append(time_per_iteration_stds, np.std(stats[i]['time_per_iteration']))
        parallel_fraction_means = np.append(parallel_fraction_means, np.mean(stats[i]['parallel_fraction']))
        parallel_fraction_stds = np.append(parallel_fraction_stds, np.std(stats[i]['parallel_fraction']))

# Sort according to number of CPUs. This makes lines in the plots nice
workers = np.array([p[0] for p in worker_perturbation_pairs])
perturbations = np.array([p[1] for p in worker_perturbation_pairs])
sort_indices = np.argsort(workers)
workers = workers[sort_indices]
perturbations = perturbations[sort_indices]
time_per_iteration_means = time_per_iteration_means[sort_indices]
time_per_iteration_stds = time_per_iteration_stds[sort_indices]
parallel_fraction_means = parallel_fraction_means[sort_indices]
parallel_fraction_stds = parallel_fraction_stds[sort_indices]

if ONLY_LOGARITHMICALLY_SPACED_WORKER_TIMES:
    keep_vals = [1,2,4,8,16,24]
    keep_ids = []
    for v in keep_vals:
        keep_ids.extend(list(np.where(workers == v)[0]))
    keep_ids = np.array(keep_ids)
    workers = workers[keep_ids]
    perturbations = perturbations[keep_ids]
    time_per_iteration_means = time_per_iteration_means[keep_ids]
    time_per_iteration_stds = time_per_iteration_stds[keep_ids]
    parallel_fraction_means = parallel_fraction_means[keep_ids]
    parallel_fraction_stds = parallel_fraction_stds[keep_ids]

# Confidence intervals
cis_times = []
for m, s in zip(time_per_iteration_means, time_per_iteration_stds):
    interval = sp.stats.norm.interval(0.95, loc=m, scale=s)
    half_width = (interval[1] - interval[0])/2
    cis_times.append(half_width)
cis_par_frac = []
for m, s in zip(parallel_fraction_means, parallel_fraction_stds):
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
    # Indices of the 1 worker job for this perturbation
    ids_minimal_workers_i_pert = np.logical_and(workers == 1, ids_perturbation)
    if not ids_minimal_workers_i_pert.any():
        continue
    # Time per iteration and of this job
    T1 = time_per_iteration_means[ids_minimal_workers_i_pert]
    T1_std = time_per_iteration_stds[ids_minimal_workers_i_pert]
    # Compute speed-ups
    TP[i_pert] = time_per_iteration_means[ids_perturbation]
    TP_std[i_pert] = time_per_iteration_stds[ids_perturbation]
    # Speed-up
    try:
        SP[i_pert] = T1/TP[i_pert]
    except ValueError:
        IPython.embed()
    SP_std[i_pert] = np.sqrt((T1_std / T1)**2 + (TP_std[i_pert] / TP[i_pert])**2) * np.abs(SP[i_pert])  # See Taylor: "Error analysis"
    # Parallel fraction
    f[i_pert] = (1 - TP[i_pert]/T1) / (1 - 1/workers_SP[i_pert])
    f[i_pert][0] = 0
    f_std[i_pert] = np.zeros(f[i_pert].shape) # TODO: Compute


# Plotting
colors = plt.cm.gnuplot(np.linspace(0, 1, 8))
matplotlib.rcParams['axes.color_cycle'] = colors

# Figure 1: Average time per iteration vs number of CPUs, logarithmic y
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = time_per_iteration_means[ids]
    s = time_per_iteration_stds[ids]
    legend.append(str(i_pert))
    ax.errorbar(x, y, yerr=s, fmt='-o')
plt.xlabel('Number of CPUs')
plt.ylabel('Average time per iteration [s]')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(legend, title='Perturbations', loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join(save_dir, Eid + '-scaling-01.pdf'), bbox_inches='tight')

# Figure 2: Average time per iteration vs number of CPUs, logarithmic y and x
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = time_per_iteration_means[ids]
    s = time_per_iteration_stds[ids]
    legend.append(str(i_pert))
    ax.errorbar(x, y, yerr=s, fmt='-o')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(legend, title='Perturbations', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Number of CPUs')
plt.ylabel('Average time per iteration [s]')
ax.set_yscale('log')
ax.set_xscale('log', basex=2)
plt.savefig(os.path.join(save_dir, Eid + '-scaling-02.pdf'), bbox_inches='tight')

# Figure 3: Average time per iteration vs number of CPUs, logarithmic y and x, smoothed interpolation
fig, ax = plt.subplots()
ax.set_prop_cycle(None)
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = time_per_iteration_means[ids]
    s = time_per_iteration_stds[ids]
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
plt.xlabel('Number of CPUs')
plt.ylabel('Time per iteration [s]')
# ax.legend(legend)
ax.set_yscale('log')
ax.set_xscale('log', basex=2)
plt.savefig(os.path.join(save_dir, Eid + '-scaling-03.pdf'), bbox_inches='tight')

# Figure 4: Amdahl's law test
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    if (workers_SP[i_pert] == 0).all():
        continue
    legend.append(str(i_pert))
    ax.errorbar(workers_SP[i_pert], SP[i_pert], yerr=SP_std[i_pert], fmt='-o')
ax.errorbar(workers_SP[np.unique(perturbations)[0]], workers_SP[np.unique(perturbations)[0]])
legend.append("Ideal")
plt.xlabel('Number of CPUs')
plt.ylabel('Speed-up factor')
ax.legend(legend, title='Perturbations', loc='upper left', ncol=2, columnspacing=0.5)
plt.savefig(os.path.join(save_dir, Eid + '-scaling-04.pdf'), bbox_inches='tight')

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
plt.savefig(os.path.join(save_dir, Eid + '-scaling-05.pdf'), bbox_inches='tight')

# Figure 6: ...
fig, ax = plt.subplots()
legend = []
for i_pert in sorted(np.unique(perturbations), reverse=False):
    ids = np.where(perturbations == i_pert)
    x = workers[ids]
    y = parallel_fraction_means[ids]
    s = parallel_fraction_stds[ids]
    legend.append(str(i_pert) + ' perturbations')
    ax.errorbar(x, y, yerr=s, fmt='-o')
plt.xlabel('Number of CPUs')
plt.ylabel('Average parallel fraction')
ax.legend(legend)
# ax.set_yscale('log')
plt.savefig(os.path.join(save_dir, Eid + '-scaling-06.pdf'), bbox_inches='tight')

# Figure 7: Amdahl's law test: Parallel fraction as function of perturbations
fig, ax = plt.subplots()
legend = []
for i_workers in sorted(np.unique(workers)):
    x = np.array([])
    y = np.array([])
    y_std = np.array([])
    for i_pert in perturbations_SP:
        idx = workers_SP[i_pert] == i_workers
        if idx.any():
            x = np.append(x, i_pert) 
            y = np.append(y, f[i_pert][idx])
            y[y<0] = 0
            y_std = np.append(y_std, f_std[i_pert][idx])
    ax.errorbar(x, y, yerr=y_std, fmt='o')
    legend.append(str(i_workers))
plt.xlabel('Number of perturbations')
plt.ylabel('Average parallel fraction')
ax.legend(legend, title='CPUs')
ax.set_xscale('log', basex=2)
# ax.set_yscale('log', basey=10)
plt.savefig(os.path.join(save_dir, Eid + '-scaling-07.pdf'), bbox_inches='tight')

copy_tree(save_dir, dst_dir)