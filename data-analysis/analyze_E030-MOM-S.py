import os
from distutils.dir_util import copy_tree
import warnings

import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch

from context import utils
import utils.filesystem as fs
import utils.plotting as plot
from utils.data_analysis import invert_signs, load_stats
from utils.misc import get_equal_dicts, length_of_longest


def create_plots(stats_list, keys_to_plot, x_key, groups, result_dir, include_val=True):
    n_keys = len(keys_to_plot)
    n_chars = len(str(n_keys))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} monitored keys plotted'
    groups_org = groups.copy()
    for i_key, k in enumerate(keys_to_plot):
        # Get data and subset only those series that are done (or the one that is the longest)
        groups = groups_org.copy()
        list_of_series = [s[k].tolist() for s in stats_list if k in s]
        list_of_xs = [s[x_key].tolist() for s in stats_list if k in s]  # [range(len(s)) for s in stats_list if k in s]  
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        groups = groups[indices]
        list_of_series = [list_of_series[i] for i in indices]
        list_of_xs = [list_of_xs[i] for i in indices]

        # Validation series
        if include_val:
            val_k = k[:-4] + '_val'
            list_of_series_val = [s[val_k].tolist() for i, s in enumerate(stats_list) if val_k in s and i in indices]
        if include_val and not len(list_of_series_val) == 0:
            # list_of_xs_val = [np.where(~np.isnan(l))[0].tolist() for l_s in list_of_series_val]
            list_of_xs_val = [np.array(l_x)[~np.isnan(l_s)].tolist() for l_x, l_s in zip(list_of_xs, list_of_series_val)]
            list_of_xs.extend(list_of_xs_val)
            list_of_series_val = [np.array(l) for l in list_of_series_val]
            list_of_series_val = [l[~np.isnan(l)].tolist() for l in list_of_series_val]
            list_of_series.extend(list_of_series_val)
            groups_val = np.array([g + ', validation' for g in groups])
            groups = np.append(groups, groups_val)

        # Sort
        list_of_xs = [x for _,x in sorted(zip(groups.tolist(), list_of_xs))]
        list_of_series = [x for _,x in sorted(zip(groups.tolist(), list_of_series))]
        groups.sort()

        # Plot
        if include_val:
            plot.timeseries_mean_grouped(list_of_xs, list_of_series, groups, xlabel=x_key, ylabel=k, map_labels='supervised')
        else:
            plot.timeseries_mean_grouped(list_of_xs, list_of_series, groups, xlabel=x_key, ylabel=k, map_labels='reinforcement')
        if include_val:
            if 'return' in k:
                plt.gca().set_ylim(0, 1.5)
            elif 'accuracy' in k:
                plt.gca().set_ylim(0.4, 1)
        if x_key == 'generations':
            plt.savefig(os.path.join(result_dir, k + '-all-series-mean-sd' + '.pdf'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(result_dir, x_key + '-' + k + '-all-series-mean-sd' + '.pdf'), bbox_inches='tight')
        plt.close()
        # Progress
        if i_key + 1 == n_keys:
            print(f.format(i_key+1, n_keys), end='\n')
        else:
            print(f.format(i_key+1, n_keys), end='\r')


def get_directories(experiment_id):
    # Get directories to analyze
    this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
    package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')
    d = os.path.join(package_root_this_file, 'experiments', 'checkpoints', experiment_id)
    directories = [os.path.join(d, di) for di in os.listdir(d) if os.path.isdir(os.path.join(d, di))]
    directories = [d for d in directories if 'monitoring' not in d and 'analysis' not in d]
    # Create result directory
    dst_dir = '/home/jakob/Dropbox/Apps/ShareLaTeX/Master\'s Thesis/graphics/' + experiment_id + '-analysis'
    result_dir = os.path.join(d, experiment_id + '-analysis')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return directories, result_dir, dst_dir


def analyze(experiment_id, keys_to_plot):
    directories, result_dir, dst_dir = get_directories(experiment_id)
    if len(directories) == 0:
        print('No results for {}'.format(experiment_id))
        return
    # Load
    stats = []
    groups = np.array([])
    for d in directories:
        try:
            st = pd.read_csv(os.path.join(d, 'stats.csv'))
            # s = torch.load(os.path.join(d, 'state-dict-algorithm.pkl'))
            with open(os.path.join(d, 'init.log'), 'r') as f:
                init = f.read()
            # Get momentum
            i = init.find('model_params')
            i = init.find('momentum', i)
            j = init.find(',', i)
            m = float(init[i+11:j])
            if m <= 0.9:
                gr_lab = r'$\gamma={}$'.format(m)
                groups = np.append(groups, gr_lab)
                stats.append(st)
        except:
            print("None in: " + d)
    # Plot
    if stats:
        invert_signs(stats)
        create_plots(stats, keys_to_plot, 'generations', groups, result_dir, include_val=True)
        # create_plots(stats, keys_to_plot, 'walltimes', groups, result_dir, include_val=True)
        copy_tree(result_dir, dst_dir)
    else:
        print('No matches for ' + experiment_id)


if __name__ == '__main__':
    # Ignore warnings from matplotlib
    warnings.filterwarnings("ignore", module="matplotlib")
    # Font setting
    matplotlib.rcParams.update({'font.size': 12})
    # Experiment IDs
    experiment_ids = ['E030-MOM-S']
    # Keys to analyze
    keys_to_plot = ['return_unp', 'return_avg', 'accuracy_unp', 'accuracy_avg']
    # Analyze
    for experiment_id in experiment_ids:
        analyze(experiment_id, keys_to_plot)
