import os
from ast import literal_eval

import IPython
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import torch
from cycler import cycler

from utils.misc import get_longest_sublists


def moving_average(y, window=100, center=True):
    """
    Compute a moving average with of `window` observations in `y`. If `centered=True`, the 
    average is computed on `window/2` observations before and after the value of `y` in question. 
    If `centered=False`, the average is computed on the `window` previous observations.
    """
    if type(y) != list:
        y = list(y)
    return pd.Series(y).rolling(window=window, center=center).mean()


def remove_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1
    ax.legend(handles, labels)


def timeseries(xdatas, ydatas, xlabel, ylabel, plotlabels=None, figsize=(6.4, 4.8)):
    if plotlabels is None:
        plotlabels = [None]*len(xdatas)
    fig = plt.figure(figsize=figsize)
    handles = []
    maxs = []
    for xdata, ydata, plotlabel in zip(xdatas, ydatas, plotlabels):
        ydata = moving_average(ydata)
        maxs.append(np.max(ydata))
        handles.extend(plt.plot(xdata, ydata, label=plotlabel))
    if plotlabels is not None:
        plt.legend(handles=handles, ncol=2, loc='best')
    # plt.gca().set_ylim([None, np.mean(maxs)*1.2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def timeseries_distribution(xdatas, ydatas, xlabel, ylabel, xbins=100, ybins=100, figsize=(6.4, 4.8)):
    plt.rcParams['image.cmap'] = 'viridis'
    # Get x and y edges spanning all values
    maxx = max([max(xdata) for xdata in xdatas])
    minx = min([min(xdata) for xdata in xdatas])
    maxy = max([max(ydata) for ydata in ydatas])
    miny = min([min(ydata) for ydata in ydatas])
    xedges = np.linspace(minx, maxx, num=xbins)
    yedges = np.linspace(miny, maxy, num=ybins)
    # Use number of bins instead if only single unique value in data
    if maxy == miny:
            yedges = ybins
    if maxx == minx:
        xedges = xbins
    H, xedges, yedges = np.histogram2d(xdatas[0], ydatas[0], bins=(xedges, yedges))
    counts = np.zeros(H.shape)
    counts += H
    for xdata, ydata in zip(xdatas[1:], ydatas[1:]):
        H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(xedges, yedges))
        counts += H
    counts = counts/counts.max()
    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(figsize=figsize)
    plt.pcolormesh(X, Y, counts.T, linewidth=0, rasterized=True)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def timeseries_median(xdatas, ydatas, xlabel, ylabel, figsize=(6.4, 4.8)):
    length = len(sorted(ydatas, key=len, reverse=True)[0])
    ydata = []
    for i in range(len(ydatas)):
        ydata.append(ydatas[i] + [np.NaN] * (length-len(ydatas[i])))
        ydata[i] = np.array(ydata[i])
    yavgs = np.nanmedian(ydata, 0)
    ymaxs = np.nanmax(ydata, 0)
    ymins = np.nanmin(ydata, 0)
    xdata = get_longest_sublists(xdatas)[0]
    h = []
    fig = plt.figure()
    h.extend(plt.plot(xdata, moving_average(yavgs), label='MA of median'))
    h.extend(plt.plot(xdata, moving_average(ymaxs), label='MA of max'))
    h.extend(plt.plot(xdata, moving_average(ymins), label='MA of min'))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=h, loc='best')


def timeseries_final_distribution(datas, label, ybins='auto', figsize=(6.4, 4.8)):
    datas_final = [ydata[-1] for ydata in datas]
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(datas_final, bins=ybins)
    plt.xlabel(label)
    plt.ylabel('Counts')


def timeseries_mean_grouped(xdatas, ydatas, groups, xlabel, ylabel, figsize=(6.4, 4.8), points_in_plot=200):
    assert type(groups) == np.ndarray
    sns.set(color_codes=True)
    plt.figure(figsize=figsize)
    legend = []
    n_groups = len(np.unique(groups))
    if n_groups <= 6:
        colors = plt.cm.inferno(np.linspace(0, 1, 6))
        colors = np.reshape(np.append(colors[0::2], colors[1::2]), (6, 4))
    else:
        colors = plt.cm.inferno(np.linspace(0, 1, n_groups))
    sns.set_style("ticks")
    for g, c in zip(set(groups), colors[0:n_groups]):
        if type(g) in [str, np.str, np.str_]:
            gstr = g
        else:
            gstr = 'G{0:02d}'.format(g)
        legend.append(gstr)
        g_indices = np.where(groups == g)[0]
        ydatas_grouped = [ydatas[i] for i in g_indices]
        length = len(sorted(ydatas_grouped, key=len, reverse=True)[0])
        nsub = int(length/points_in_plot)
        ydata = np.array([ydata+[np.NaN]*(length-len(ydata)) for ydata in ydatas_grouped])
        # Subsample y
        ydata_subsampled = ydata[:,::nsub] if np.prod(ydata.shape) > nsub else ydata
        # if ydata_subsampled[-1] != ydata[-1]:
        #     ydata_subsampled = np.append(ydata_subsampled, ydata[0,-1])
        x = [xdatas[i] for i in g_indices]
        x = get_longest_sublists(x)[0]
        if type(x) in [range, list]:
            x = np.array(x)
        # Subsample x
        x_subsampled = x[::nsub] if len(x) > nsub else x
        # if x_subsampled[-1] != x[-1] 
        #     x_subsampled = np.append(x_subsampled, x[-1])
        ax = sns.tsplot(value=ylabel, data=ydata_subsampled, time=x_subsampled, ci="sd", estimator=np.mean, color=c)
    lines = list(filter(lambda c: type(c)==mpl.lines.Line2D, ax.get_children()))
    plt.legend(handles=lines, labels=legend)
    plt.xlabel(xlabel)


def load_stats(stats_file):
    stats = pd.read_csv(stats_file)
    for k in stats.keys()[stats.dtypes == object]:
        try:
            stats[k] = stats[k].apply(literal_eval)
        except:
            pass
    return stats


def plot_stats(stats_file, chkpt_dir, wide_figure=True):
    """
    Plots training statistics
    - Unperturbed return
    - Average return
    - Maximum return
    - Minimum return
    - Smoothed version of the above
    - Return variance
    - Rank of unperturbed model
    - Sigma
    - Learning rate
    - Total wall clock time
    - Wall clock time per generation

    Possible x-axes are:
    - Generations
    - Episodes
    - Observations
    - Walltimes
    """

    # Plot settings
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    if wide_figure:
        figsize = mpl.figure.figaspect(9/16)
    else:
        figsize = (6.4, 4.8)

    # Load data
    try:
        stats = load_stats(stats_file)
    except:
        return

    # Invert sign on negative returns (negative returns indicate a converted minimization problem)
    if (np.array(stats['return_max']) < 0).all():
        for k in ['return_unp', 'return_avg', 'return_min', 'return_max', 'return_val']:
            stats[k] = [-s for s in stats[k]]

    # Computations/Transformations
    psuedo_start_time = stats['walltimes'].diff().mean()
    # Add pseudo start time to all times
    abs_walltimes = stats['walltimes'] + psuedo_start_time
    # Append pseudo start time to top of series and compute differences
    stats['time_per_generation'] = pd.concat([pd.Series(psuedo_start_time), abs_walltimes]).diff().dropna()
    stats['parallel_fraction'] = stats['workertimes']/stats['time_per_generation']

    # Compute moving averages
    for c in stats.columns:
        if not 'Unnamed' in c and c[-3:] != '_ma':
            stats[c + '_ma'] = stats[c].rolling(window=100, center=True, win_type=None).mean()
        
    # Plot each of the columns including moving average
    c_list = stats.columns.tolist()
    while c_list:
        c = c_list.pop()
        is_unnamed = lambda c: 'Unnamed' in c
        is_part_of_multi_series = lambda c: c.split('_')[-1].isdigit()
        is_moving_average = lambda c: c[-3:] == '_ma'
        if not is_unnamed(c) and not is_moving_average(c):
            if is_part_of_multi_series(c):
                # Find all c that are in this series
                cis = {ci for ci in stats.columns if ci.split('_')[:-1] == c.split('_')[:-1] and not is_moving_average(c)}
                for ci in cis.difference({c}):
                    c_list.remove(ci)
                cis = sorted(list(cis))
                c = ''.join(c.split('_')[:-1])
                # Loop over them and plot into same plot
                fig, ax = plt.subplots(figsize=figsize)
                for ci in cis:
                    stats[ci].plot(ax=ax, linestyle='None', marker='.', alpha=0.2, label='_nolegend_')
                ax.set_prop_cycle(None)
                for ci in cis:
                    stats[ci + '_ma'].plot(ax=ax, linestyle='-', label=ci)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                fig, ax = plt.subplots(figsize=figsize)
                stats[c].plot(ax=ax, alpha=0.2, linestyle='None', marker='.', label='_nolegend_')
                ax.set_prop_cycle(None)
                stats[c + '_ma'].plot(ax=ax, linestyle='-', label='_nolegend_')
                # ax.legend(loc='best')
            plt.xlabel('Iteration')
            plt.ylabel(c)
            fig.savefig(os.path.join(chkpt_dir, c + '.pdf'), bbox_inches='tight')
            plt.close(fig)
