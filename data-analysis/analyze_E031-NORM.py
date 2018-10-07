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


def get_directories(experiment_id):
    # Get directories to analyze
    this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
    package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')
    d = os.path.join(package_root_this_file, 'experiments', 'checkpoints', experiment_id)
    directories = [os.path.join(d, di) for di in os.listdir(d) if os.path.isdir(os.path.join(d, di))]
    directories = [d for d in directories if 'monitoring' not in d and 'analysis' not in d]
    # Create result directory
    # dst_dir = '/Users/Jakob/Dropbox/Apps/ShareLaTeX/Master\'s Thesis/graphics/' + experiment_id + '-analysis'
    dst_dir = '/home/jakob/Dropbox/Apps/ShareLaTeX/Master\'s Thesis/graphics/' + experiment_id + '-analysis'
    result_dir = os.path.join(d, experiment_id + '-analysis')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return directories, result_dir, dst_dir


def get_data(experiment_id):
    directories, result_dir, dst_dir = get_directories(experiment_id)
    if len(directories) == 0:
        print('No results for {}'.format(experiment_id))
        return
    # Load
    stats = []
    groups = np.array([])
    g1 = g2 = g3 = g4 = 0
    for d in directories:
        try:
            st = pd.read_csv(os.path.join(d, 'stats.csv'))
            s = torch.load(os.path.join(d, 'state-dict-algorithm.pkl'))
            gr_lab = None
            if s['optimize_sigma'] is None:
                g1 += 1
                gr_lab = 'isotropic-fixed-' + str(g1)
            elif s['optimize_sigma'] == 'single':
                g2 += 1
                gr_lab = 'isotropic-adapted-' + str(g2)
            elif s['optimize_sigma'] == 'per-layer':
                g3 += 1
                gr_lab = 'separable-layer-' + str(g3)
            elif s['optimize_sigma'] == 'per-weight':
                g4 += 1
                gr_lab = 'separable-parameter-' + str(g4)
            else:
                raise ValueError("Unkown `optimize_sigma` value")
            if gr_lab is not None:
                groups = np.append(groups, gr_lab)
                stats.append(st)
        except:
            print("None in: " + d)
    if stats:
        invert_signs(stats)
    return stats, groups, result_dir, dst_dir


def plot_variance_single(s, g, result_dir):
    # Variance
    sigma_label = r'$\sigma$'
    fig, ax = plt.subplots()
    s['sigma'].plot()
    ax.set_xlabel('Iteration')
    ax.set_ylabel(sigma_label)
    fig.savefig(os.path.join(result_dir, g + '-variance.pdf'), bbox_inches='tight')
    plt.close(fig)

    # All together
    if g[:17] == 'isotropic-adapted':
        legend_location = 'best'
    else:
        legend_location = 'lower right'
    grad_label = r'$\Vert\nabla_{\mathbf{w}}U(\boldsymbol{\mu},\sigma)\Vert$'
    param_label = r'$\Vert\boldsymbol{\mu}\Vert$'
    sigma_label = r'$100\times\sigma$'
    fig, ax = plt.subplots()
    s['grad_norm'].plot(ax=ax, color='tab:blue', linestyle='None', marker='.', alpha=0.1, label='_nolegend_')
    s['grad_norm_mean'].plot(ax=ax, color='tab:blue', label=grad_label)
    s['param_norm'].plot(ax=ax, secondary_y=True, color='tab:orange', label=param_label)
    (s['sigma'] * 100).plot(ax=ax, color='tab:green', label=sigma_label)
    lines = ax.get_lines() + ax.right_ax.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc=legend_location)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(grad_label + ' and ' + sigma_label)
    ax.right_ax.set_ylabel(param_label)
    fig.savefig(os.path.join(result_dir, g + '-param-and-grad-and-variance-norm.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_variance_layer(s, g, result_dir):
    # IPython.embed()
    # Get keys and colors
    keys = []
    sigma_labels = []
    for k in s:
        if k.startswith('sigma'):
            keys.append(k)
            n = k[6:]
            sigma_labels.append('$\\sigma_{' + '{n}' + '}$')
            # sigma_labels.append('$\sigma_{}$'.format(n))
    colors = plt.cm.tab20(np.linspace(0, 1, len(keys)))

    # Variance
    fig, ax = plt.subplots()
    for k, l, c in zip(keys, sigma_labels, colors):
        s[k].plot(ax=ax, label=l, c=c)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\sigma$')
    ax.legend(ax.get_lines(), [l.get_label() for l in ax.get_lines()], loc=2, ncol=7, borderaxespad=.2, mode="expand")
    fig.savefig(os.path.join(result_dir, g + '-variance.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_variance_parameter(s, g, result_dir):
    sigmas = ['sigma_min', 'sigma_max', 'sigma_avg', 'sigma_med']
    key_label_map = {'sigma_min': r'$\sigma_\text{min}$',
                     'sigma_max': r'$\sigma_\text{max}$',
                     'sigma_avg': r'$\sigma_\text{avg}$',
                     'sigma_med': r'$\sigma_\text{med}$'}
    fig, ax = plt.subplots()
    for k in sigmas:
        s[k].plot(ax=ax, label=key_label_map[k])
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\sigma$')
    ax.set_yscale('log')
    ax.legend(ax.get_lines(), [l.get_label() for l in ax.get_lines()], loc='upper left')
    fig.savefig(os.path.join(result_dir, g + '-variance.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_norms(s, g, result_dir):
    # # Gradient norm
    # fig, ax = plt.subplots()
    # create_norm_plot(ax, s, 'grad_norm')
    # fig.savefig(os.path.join(result_dir, g + '-grad-norm.pdf'), bbox_inches='tight')
    # plt.close(fig)

    # # Parameter norm
    # fig, ax = plt.subplots()
    # create_norm_plot(ax, s, 'param_norm')
    # fig.savefig(os.path.join(result_dir, g + '-param-norm.pdf'), bbox_inches='tight')
    # plt.close(fig)

    # In same plot
    fig, ax = plt.subplots()
    grad_label = r'$\Vert\nabla_{\mathbf{w}}U(\boldsymbol{\mu},\sigma)\Vert$'
    param_label = r'$\Vert\boldsymbol{\mu}\Vert$'
    fig, ax = plt.subplots()
    s['grad_norm'].plot(ax=ax, color='tab:blue', linestyle='None', marker='.', alpha=0.1, label='_nolegend_')
    s['grad_norm_mean'].plot(ax=ax, color='tab:blue', label=grad_label)
    s['param_norm'].plot(ax=ax, secondary_y=True, color='tab:orange', label=param_label)
    lines = ax.get_lines() + ax.right_ax.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(grad_label)
    ax.right_ax.set_ylabel(param_label)
    fig.savefig(os.path.join(result_dir, g + '-param-and-grad-norm.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot(s, g, result_dir):
    plot_norms(s, g, result_dir)
    if 'sigma' not in s:
        if 'sigma_min' in s:
            plot_variance_parameter(s, g, result_dir)
        elif 'sigma_0' in s:
            plot_variance_layer(s, g, result_dir)
        else:
            raise ValueError("Could not find `sigma` key")
    else:
        plot_variance_single(s, g, result_dir)


if __name__ == '__main__':
    # Ignore warnings from matplotlib
    warnings.filterwarnings("ignore", module="matplotlib")
    # Font setting
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    # Experiment IDs
    experiment_id = 'E031-NORM'
    # Variance learning rates are increase through S-S3. In S3 5000 iterations are done
    # Analyze
    stats, groups, result_dir, dst_dir = get_data(experiment_id)
    for s in stats:
        s['grad_norm_mean'] = s['grad_norm'].rolling(window=50).mean()

    for i, (s, g) in enumerate(zip(stats, groups)):
        plot(s, g, result_dir)
        print('{}/{}'.format(i+1, len(stats)))

    copy_tree(result_dir, dst_dir)
