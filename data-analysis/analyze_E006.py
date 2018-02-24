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
    args.d = "/home/jakob/Dropbox/es-rl/experiments/checkpoints/E006-LVBS"

directories = [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']
filename = 'state-dict-algorithm.pkl'
rm_filenames = ['state-dict-best-algorithm.pkl', 'state-dict-best-optimizer.pkl', 'state-dict-best-model.pkl']

algorithm_states = []
for d in directories:
    try:
        algorithm_states.append(torch.load(os.path.join(d, filename)))
    except:
        pass
    # for rmf in rm_filenames:
    #     try:
    #         IPython.embed()
    #         os.remove(os.path.join(d, rmf))
    #     except OSError:
    #         pass

# Get groups
keys_to_ignore = set(algorithm_states[0].keys())
keys_to_ignore.remove('batch_size')
groups = get_equal_dicts(algorithm_states, ignored_keys=keys_to_ignore)

# Get data
n_batch_sizes = len(algorithm_states)
n_generations = algorithm_states[0]['max_generations']
batch_sizes = np.array([s['batch_size'] for s in algorithm_states])
losses = [np.array(s['stats']['return_unp']) for s in algorithm_states]
stds = np.array([np.std(l) for l in losses])
times = np.array([s['stats']['walltimes'][-1] for s in algorithm_states])
abs_walltimes = [np.array(s['stats']['walltimes']) + s['stats']['start_time'] for s in algorithm_states]
generation_times = [np.diff(np.array([s['stats']['start_time']] + list(awt))) for s, awt in zip(algorithm_states, abs_walltimes)]
parallel_fractions = np.array([np.mean(np.array(s['stats']['workertimes'])/gt) for s, gt in zip(algorithm_states, generation_times)])

# Compute mean and std over groups
mean_of_stds_per_bs = np.array([])
std_of_stds_per_bs = np.array([])
mean_of_times_per_bs = np.array([])
std_of_times_per_bs = np.array([])
mean_of_parallel_fractions = np.array([])
std_of_parallel_fractions = np.array([])
associated_bs = np.array([])
for g_id in set(groups):
    ids = np.where(groups == g_id)[0]
    mean_of_stds_per_bs = np.append(mean_of_stds_per_bs, np.mean(stds[ids]))
    std_of_stds_per_bs = np.append(std_of_stds_per_bs, np.std(stds[ids]))
    mean_of_times_per_bs = np.append(mean_of_times_per_bs, np.mean(times[ids]))
    std_of_times_per_bs = np.append(std_of_times_per_bs, np.std(times[ids]))
    mean_of_parallel_fractions = np.append(mean_of_parallel_fractions, np.mean(parallel_fractions[ids]))
    std_of_parallel_fractions = np.append(std_of_parallel_fractions, np.std(parallel_fractions[ids]))
    associated_bs = np.append(associated_bs, batch_sizes[ids[0]])
mean_of_times_per_bs = mean_of_times_per_bs/n_generations

# Confidence intervals
cis = []
for m, s, b in zip(mean_of_stds_per_bs, std_of_stds_per_bs, associated_bs):
    interval = sp.stats.norm.interval(0.95, loc=m, scale=s)
    half_width = (interval[1] - interval[0])/2
    cis.append(half_width)

# Get linear and logarithmic data parts for pretty plotting
# I know this should have been [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384] but got the other data
log_bs = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048,4192,8384,16768])
log_ids = np.isin(associated_bs, log_bs)

associated_bs_linear = associated_bs[~log_ids]
mean_of_stds_per_bs_linear = mean_of_stds_per_bs[~log_ids]
std_of_stds_per_bs_linear = std_of_stds_per_bs[~log_ids]
cis_linear = [ci for i, ci in enumerate(cis) if not log_ids[i]]

associated_bs_log = associated_bs[log_ids]
mean_of_stds_per_bs_log = mean_of_stds_per_bs[log_ids]
std_of_stds_per_bs_log = std_of_stds_per_bs[log_ids]
cis_log = [ci for i, ci in enumerate(cis) if log_ids[i]]

# Plotting
fig, ax = plt.subplots()
# bars = ax.errorbar(associated_bs_linear, mean_of_stds_per_bs_linear, yerr=cis_linear, fmt='o', ecolor='g')
bars = ax.errorbar(associated_bs_linear, mean_of_stds_per_bs_linear, yerr=std_of_stds_per_bs_linear, fmt='o')
plt.xlabel('Batch size')
plt.ylabel('Loss function variance')
plt.savefig(os.path.join(args.d,'E006-loss-variance-vs-batch-size-01.pdf'), bbox_inches='tight')

fig, ax = plt.subplots()
ax.errorbar(associated_bs_log, mean_of_stds_per_bs_log, yerr=std_of_stds_per_bs_log, fmt='o')
plt.xlabel('Batch size')
plt.ylabel('Loss function variance')
ax.set_xscale('log')
plt.savefig(os.path.join(args.d,'E006-loss-variance-vs-batch-size-02.pdf'), bbox_inches='tight')

fig, ax = plt.subplots()
ax.errorbar(associated_bs, mean_of_stds_per_bs, yerr=std_of_stds_per_bs, fmt='o')
plt.xlabel('Batch size')
plt.ylabel('Loss function variance')
ax.set_xscale('log')
plt.savefig(os.path.join(args.d,'E006-loss-variance-vs-batch-size-03.pdf'), bbox_inches='tight')

# Labels for Pareto plot
bss = [1, 4, 10, 64, 256, 1024, 2048]
xy = []
text = []
for bs in bss:
    i = np.where(associated_bs == bs)
    xy.append((mean_of_times_per_bs[i], mean_of_stds_per_bs[i]))
    text.append(str(int(bs)))


fig, ax = plt.subplots()
ax.errorbar(mean_of_times_per_bs, mean_of_stds_per_bs, xerr=std_of_times_per_bs, yerr=std_of_stds_per_bs, fmt='.')
plt.xlabel('Time per generation')
plt.ylabel('Loss function variance')
plt.title('Pareto front')
ax.set_xscale('log')
for t, (x, y) in zip(text, xy):
    ax.annotate(t, xy=(x, y), xytext=(x*1.1, y*1.1))
plt.savefig(os.path.join(args.d,'E006-loss-variance-vs-batch-size-04.pdf'), bbox_inches='tight')

fig, ax = plt.subplots()
ax.errorbar(mean_of_parallel_fractions, mean_of_stds_per_bs, xerr=std_of_parallel_fractions, yerr=std_of_stds_per_bs, fmt='.')
plt.xlabel('Parallel fraction')
plt.ylabel('Loss function variance')
plt.title('Pareto front')
for t, (x, y) in zip(text, xy):
    ax.annotate(t, xy=(x, y), xytext=(x*1.1, y*1.1))
plt.savefig(os.path.join(args.d,'E006-loss-variance-vs-batch-size-05.pdf'), bbox_inches='tight')


IPython.embed()
