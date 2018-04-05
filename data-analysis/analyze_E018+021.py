import os

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import torch
from context import utils
import utils.filesystem as fs
import utils.plotting as plot
from utils.misc import get_equal_dicts, length_of_longest
from data_analysis import load_stats, invert_signs


def create_plots(stats_list, keys_to_monitor, groups):
    unique_groups = set(groups)
    n_keys = len(keys_to_monitor)
    n_chars = len(str(n_keys))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} monitored keys plotted'
    for i_key, k in enumerate(keys_to_monitor):
        list_of_series = [s[k].tolist() for s in stats_list if k in s]
        list_of_genera = [range(len(s)) for s in stats_list if k in s]

        plot.timeseries(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(analysis_dir, k + '-all-series.pdf'), bbox_inches='tight')
        plt.close()

        plot.timeseries_distribution(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(analysis_dir, k + '-all-distribution.pdf'), bbox_inches='tight')
        plt.close()

        plot.timeseries_median(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(analysis_dir, k + '-all-median.pdf'), bbox_inches='tight')
        plt.close()

        plot.timeseries_final_distribution(list_of_series, label=k, ybins=len(list_of_series)*10)
        plt.savefig(os.path.join(analysis_dir, k + '-all-final-distribution.pdf'), bbox_inches='tight')
        plt.close()

        # Subset only those series that are done (or the one that is the longest)
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        list_of_longest_series = [list_of_series[i] for i in indices]
        list_of_longest_genera = [list_of_genera[i] for i in indices]
        groups_longest_series = groups[indices]
        plot.timeseries_mean_grouped(list_of_longest_genera, list_of_longest_series, groups_longest_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(analysis_dir, k + '-all-series-mean-sd' + '.pdf'), bbox_inches='tight')
        plt.close()

        if len(unique_groups) > 1:
            for g in unique_groups:
                if type(g) in [str, np.str, np.str_]:
                    gstr = g
                else:
                    gstr = 'G{0:02d}'.format(g)
                g_indices = np.where(groups == g)[0]
                group_stats = [stats_list[i] for i in g_indices]

                list_of_series = [s[k].tolist() for s in group_stats if k in s]
                list_of_genera = [range(len(s)) for s in group_stats if k in s]
                if list_of_genera and list_of_series:
                    plot.timeseries(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                    plt.savefig(os.path.join(analysis_dir, k + '-group-' + gstr + '-series.pdf'), bbox_inches='tight')
                    plt.close()

                    plot.timeseries_distribution(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                    plt.savefig(os.path.join(analysis_dir, k + '-group-' + gstr + '-distribution.pdf'), bbox_inches='tight')
                    plt.close()

                    plot.timeseries_median(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                    plt.savefig(os.path.join(analysis_dir, k + '-group-' + gstr + '-median.pdf'), bbox_inches='tight')
                    plt.close()

                    plot.timeseries_final_distribution(list_of_series, label=k, ybins=len(list_of_series)*10)
                    plt.savefig(os.path.join(analysis_dir, k + '-group-' + gstr + '-final-distribution.pdf'), bbox_inches='tight')
                    plt.close()

        if i_key + 1 == n_keys:
            print(f.format(i_key+1, n_keys), end='\n')
        else:
            print(f.format(i_key+1, n_keys), end='\r')

# Data directories
i = 'E014-SM'
keys_to_plot = {'return_unp', 'return_avg', 'accuracy_unp', 'accuracy_avg'}
this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')
d = os.path.join(package_root_this_file, 'experiments', 'checkpoints', i)
directories = [os.path.join(d, di) for di in os.listdir(d) if os.path.isdir(os.path.join(d, di)) and di != 'monitoring']
analysis_dir = os.path.join(d, str(i) + '-analysis')
if not os.path.exists(analysis_dir):
    os.mkdir(analysis_dir)

# Load
stats = []
groups = np.array([])
for d in directories:
    try:
        s = torch.load(os.path.join(d, 'state-dict-best-algorithm.pkl'))
        if s['safe_mutation'] == 'SUM':
            groups = np.append(groups, 'SM-G-SUM')
        else:
            groups = np.append(groups, 'No SM')
        stats.append(pd.read_csv(os.path.join(d, 'stats.csv')))
    except:
        print("None in: " + d)

# Plot
invert_signs(stats)
create_plots(stats, keys_to_plot, groups)


# sns.set(color_codes=True)
# plt.figure(figsize=figsize)
# legend = []
# colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(groups))))

# ydatas = [s['']]

# ax = sns.tsplot(value=ylabel, data=ydata_subsampled, time=x_subsampled, ci="sd", estimator=np.mean, color=c)
