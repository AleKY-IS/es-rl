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
from utils.data_analysis import invert_signs, load_stats
from utils.misc import get_equal_dicts, length_of_longest
from utils.data_analysis import lookup_label


def create_plots(stats_list, keys_to_plot, x_key, groups, result_dir, include_val=True):
    n_keys = len(keys_to_plot)
    n_chars = len(str(n_keys))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} monitored keys plotted'
    groups_org = groups.copy()
    for i_key, k in enumerate(keys_to_plot):
        # Get data and subset only those series that are done (or the one that is the longest)
        # list_of_xs = [s[x_key].tolist() for s in stats_list if k in s]
        # l = length_of_longest(list_of_xs)
        # indices = [i for i, series in enumerate(list_of_xs) if len(series) == l]
        # groups = groups[indices]
        # list_of_xs = [list_of_xs[i] for i in indices]

        k = k[:-4] + '_val'  # Overwrite with validation key
        groups = groups_org.copy()
        list_of_series = [s[k] for s in stats_list]
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        list_of_series = [list_of_series[i] for i in indices]
        groups = groups_org[indices]
        xs = np.where(~list_of_series[0].isna())[0]
        list_of_xs = [xs for i in range(len(list_of_series))]
        list_of_series = [l[np.where(~l.isna())[0]] for l in list_of_series]
        list_of_series = [l.tolist() for l in list_of_series]
        # IPython.embed()



        # groups = groups_org.copy()
        # list_of_series = [s[k].tolist() for s in stats_list if k in s]
        # list_of_xs = [s[x_key].tolist() for s in stats_list if k in s]  # [range(len(s)) for s in stats_list if k in s]  
        # l = length_of_longest(list_of_series)
        # indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        # groups = groups[indices]
        # list_of_series = [list_of_series[i] for i in indices]
        # list_of_xs = [list_of_xs[i] for i in indices]

        # # Validation series
        # if include_val:
        #     val_k = k[:-4] + '_val'
        #     list_of_series_val = [s[val_k].tolist() for i, s in enumerate(stats_list) if val_k in s and i in indices]
        # if include_val and not len(list_of_series_val) == 0:
        #     # list_of_xs_val = [np.where(~np.isnan(l))[0].tolist() for l_s in list_of_series_val]
        #     list_of_xs_val = [np.array(l_x)[~np.isnan(l_s)].tolist() for l_x, l_s in zip(list_of_xs, list_of_series_val)]
        #     list_of_xs.extend(list_of_xs_val)
        #     list_of_series_val = [np.array(l) for l in list_of_series_val]
        #     list_of_series_val = [l[~np.isnan(l)].tolist() for l in list_of_series_val]
        #     list_of_series.extend(list_of_series_val)
        #     groups_val = np.array([g + ', validation' for g in groups])
        #     groups = np.append(groups, groups_val)

        

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
                plt.gca().set_ylim(0, 0.4)
            elif 'accuracy' in k:
                plt.gca().set_ylim(0.85, 1)
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

def final_distribution(stats, keys_to_plot, groups, result_dir, include_val=True):
    groups_org = groups.copy()
    include_val_setting = include_val
    for i_key, k in enumerate(keys_to_plot):
        include_val = include_val_setting
        if include_val:
            for s in stats:
                if k[-4:] != '_unp' or k[:-4] + '_val' not in s:
                    include_val = False
        # Get data and subset only those series that are done (or the one that is the longest)
        list_of_series = [s[k].tolist() for s in stats if k in s]
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        groups = groups_org[indices]
        list_of_final = [list_of_series[i][-1] for i in indices]
        # 
        fig, ax = plt.subplots()
        xlabel = lookup_label(k, mode='supervised')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('CDF')
        legend = []
        n_groups = len(np.unique(groups))
        df = pd.DataFrame([])
        colors = plt.cm.gnuplot(np.linspace(0, 1, n_groups))
        for g, c in zip(np.unique(groups), colors[0:n_groups]):
            g_indices = np.where(groups == g)[0]
            list_of_final_group = [list_of_final[i] for i in g_indices]
            ax.hist(list_of_final_group, alpha=0.6, density=True, histtype='step', cumulative=True, linewidth=2, color=c)
            legend.append(g)
            df_new = pd.DataFrame({g: list_of_final_group})
            df = pd.concat([df, df_new], axis=1) 
        ax.legend(legend, loc='northwest')
        fig.savefig(os.path.join(result_dir, k + '-final-distribution' + '.pdf'), bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots()
        my_order = [r'Isotropic (fixed $\sigma$)', r'Isotropic', r'Separable (layer)', r'Separable (parameter)']
        positions = [my_order.index(e) for e in df.keys() if e in my_order]
        df.boxplot(rot=10, positions=positions, showfliers=True)
        ax.xaxis.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel(xlabel)
        # ax.set_ylim(auto=True)
        fig.savefig(os.path.join(result_dir, k + '-final-distribution-boxplot' + '.pdf'), bbox_inches='tight')
        plt.close(fig)

        #
        if include_val:
            k_val = k[:-4] + '_val'
            list_of_series = [s[k_val].tolist() for s in stats if k_val in s]
            l = length_of_longest(list_of_series)
            indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
            groups = groups_org[indices]
            list_of_final = []
            for i in indices:
                a = np.array(list_of_series[i])
                list_of_final.append(a[~np.isnan(a)][-1])
            #
            fig, ax = plt.subplots()
            xlabel = lookup_label(k_val, mode='supervised')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('CDF')
            legend = []
            n_groups = len(np.unique(groups))
            df = pd.DataFrame([])
            colors = plt.cm.gnuplot(np.linspace(0.1, 1, n_groups))
            for g, c in zip(np.unique(groups), colors[0:n_groups]):
                g_indices = np.where(groups == g)[0]
                list_of_final_group = [list_of_final[i] for i in g_indices]
                ax.hist(list_of_final_group, alpha=0.6, density=True, histtype='step', cumulative=True, linewidth=2, color=c)
                legend.append(g)
                df_new = pd.DataFrame({g: list_of_final_group})
                df = pd.concat([df, df_new], axis=1)
            fig.savefig(os.path.join(result_dir, k_val + '-final-distribution' + '.pdf'), bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots()
            my_order = [r'Isotropic (fixed $\sigma$)', r'Isotropic', r'Separable (layer)', r'Separable (parameter)']
            positions = [my_order.index(e) for e in df.keys() if e in my_order]
            df.boxplot(rot=10, positions=positions, showfliers=True)
            ax.xaxis.grid(False)
            ax.set_xlabel('')
            ax.set_ylabel(xlabel)
            fig.savefig(os.path.join(result_dir, k_val + '-final-distribution-boxplot' + '.pdf'), bbox_inches='tight')
            plt.close(fig)


def violinplots(stats, keys_to_plot, groups, result_dir):
    groups_org = groups.copy()

    if keys_to_plot[0][:-4] == 'return':
        ylabel = 'NLL'
    elif keys_to_plot[0][:-4] == 'accuracy':
        ylabel = 'Classification accuracy'

    df = pd.DataFrame([])
    for k in keys_to_plot:
        # Get data and subset only those series that are done (or the one that is the longest)
        # list_of_series = [s[k].tolist() for s in stats if k in s]
        # l = length_of_longest(list_of_series)
        # indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        # groups = groups_org[indices]
        # list_of_final = [list_of_series[i][-1] for i in indices]
        list_of_series = [s[k].tolist() for s in stats if k in s]
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        groups = groups_org[indices]
        list_of_final = []
        for i in indices:
            a = np.array(list_of_series[i])
            list_of_final.append(a[~np.isnan(a)][-1])
        # 
        n_groups = len(np.unique(groups))
        colors = plt.cm.gnuplot(np.linspace(0, 1, n_groups))
        for g, c in zip(np.unique(groups), colors[0:n_groups]):
            g_indices = np.where(groups == g)[0]
            list_of_final_group = [list_of_final[i] for i in g_indices]
            label = ' '.join(lookup_label(k, mode='supervised').split(' ')[:1]) # Only two first label words (disregard accucracy, NLL etc.)
            if k[-4:] == '_val':
                label = 'Validation (unperturbed)'
            df_new = pd.DataFrame({'final_val': list_of_final_group, 'group': g, 'label': label})
            df = pd.concat([df, df_new], axis=0, ignore_index=True) 

    # my_order = [r'Isotropic (fixed $\sigma$)', r'Isotropic', r'Separable (layer)', r'Separable (parameter)']
    # positions = [my_order.index(e) for e in df.keys() if e in my_order]

    fig, ax = plt.subplots()
    fig.set_size_inches(*plt.rcParams.get('figure.figsize'))
    g = sns.factorplot(ax=ax, x="group", y="final_val", hue="label", data=df, kind="violin", legend=False)
    g.despine(left=True)
    # g.set_xticklabels(rotation=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)
    fig.savefig(os.path.join(result_dir, k[:-4] + '-final-distribution-boxplot-grouped' + '.pdf'), bbox_inches='tight')
    plt.close(fig)   


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


def get_data(experiment_id, keys_to_plot):
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
            s = torch.load(os.path.join(d, 'state-dict-algorithm.pkl'))

            gr_lab = None
            if s['optimize_sigma'] is None:
                gr_lab  = r'Isotropic (fixed $\sigma$)'
            else:
                with open(os.path.join(d, 'init.log'), 'r') as f:
                    init = f.read()
                i = init.find('_beta')
                i = init.find('lr', i)
                j = init.find(',', i)
                lr = float(init[i+5:j])
                if s['optimize_sigma'] == 'single' and lr == 2.0:  # One run with 3.0 (unconverged) and one with 2.0
                    gr_lab = r'Isotropic'
                if s['optimize_sigma'] == 'per-layer':
                    gr_lab  = r'Separable (layer)'
                if s['optimize_sigma'] == 'per-weight':
                    gr_lab  = r'Separable (parameter)'
            if gr_lab is not None:
                groups = np.append(groups, gr_lab)
                stats.append(st)
        except:
            print("None in: " + d)
    if experiment_id == 'E029-VO-S3':
        stats = [s[0:3000] for s in stats]
    if experiment_id == 'E029-VO-S5-MD':
        for g, s in zip(groups, stats):
            if g  == r'Separable (parameter)':
                s.loc[s['return_avg'] < -10, 'return_avg'] = np.nan
                s['return_avg'].fillna(method='ffill', inplace=True)
    if stats:
        invert_signs(stats)
    return stats, groups, result_dir, dst_dir


if __name__ == '__main__':
    # Ignore warnings from matplotlib
    warnings.filterwarnings("ignore", module="matplotlib")
    # Font setting
    matplotlib.rcParams.update({'font.size': 12})
    # Experiment IDs
    experiment_ids = ['E029-VO-S5-MD', 'E029-VO-S4-MD', 'E029-VO-S3', 'E029-VO-S', 'E029-VO-S2']
    # Variance learning rates are increase through S-S3. In S3 5000 iterations are done
    # Keys to analyze
    keys_to_plot = ['return_unp', 'return_avg', 'accuracy_unp', 'accuracy_avg']
    # Analyze
    for experiment_id in experiment_ids:
        stats, groups, result_dir, dst_dir = get_data(experiment_id, keys_to_plot)
        if stats:
            final_distribution(stats, keys_to_plot, groups, result_dir, include_val=True)

    import seaborn as sns
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    for experiment_id in experiment_ids:
        stats, groups, result_dir, dst_dir = get_data(experiment_id, keys_to_plot)
        if stats:
            violinplots(stats, ['return_unp','return_avg','return_val'], groups, result_dir)
            violinplots(stats, ['accuracy_unp','accuracy_avg','accuracy_val'], groups, result_dir)

    import utils.plotting as plot
    for experiment_id in experiment_ids:
        stats, groups, result_dir, dst_dir = get_data(experiment_id, keys_to_plot)
        if stats:
            create_plots(stats, keys_to_plot, 'generations', groups, result_dir, include_val=True)
        copy_tree(result_dir, dst_dir)
