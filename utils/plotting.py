import os

import IPython
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd

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


def subsample(data, axis=0, keep_percentage=10):
    pass

# def timeseries_single(xdata, ydata, xlabel, ylabel, plotlabel=None):
#     fig = plt.figure()
#     ydata = moving_average(ydata)
#     plt.plot(xdata, ydata, label=plotlabel)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)


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
    ydata = np.array([ydata+[np.NaN]*(length-len(ydata)) for ydata in ydatas])
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


def timeseries_median_grouped(xdatas, ydatas, groups, xlabel, ylabel, figsize=(6.4, 4.8)):
    assert type(groups) == np.ndarray
    sns.set(color_codes=True)
    plt.figure(figsize=figsize)
    legend = []
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for g in set(groups):
        gstr = 'G{0:02d}'.format(g)
        legend.append(gstr)
        g_indices = np.where(groups == g)[0]
        ydatas_grouped = [ydatas[i] for i in g_indices]
        length = len(sorted(ydatas_grouped, key=len, reverse=True)[0])
        ydata = np.array([ydata+[np.NaN]*(length-len(ydata)) for ydata in ydatas_grouped])
        ydata_subsampled = ydata  # ydata[:,::100]
        x = [xdatas[i] for i in g_indices]
        x = get_longest_sublists(x)[0]
        x_subsampled = x  # x[::100]
        ax = sns.tsplot(value=ylabel, data=ydata_subsampled, time=x_subsampled, ci="sd", estimator=np.mean, color=colors[g])
    lines = list(filter(lambda c: type(c)==mpl.lines.Line2D, ax.get_children()))
    plt.legend(handles=lines, labels=legend)
    plt.xlabel(xlabel)


def plot_stats(stats, chkpt_dir):
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
    figsize = (6.4, 4.8)
    pstats = stats.copy()
    x = 'generations'
    back_alpha = 0.3

    # Invert sign on negative returns (negative returns indicate a converted minimization problem)
    if (np.array(pstats['return_max']) < 0).all():
        for k in ['return_unp', 'return_avg', 'return_min', 'return_max']:
            pstats[k] = [-s for s in pstats[k]]
    
    # Only consider the first parameter group of the optimizer
    n_groups = 1 if type(pstats['lr'][0]) is float else len(pstats['lr'][0])
    if n_groups > 1:
        for key in ['lr']:
            pstats[key] = [vals_group[0] for vals_group in pstats[key]]

    # Computations/Transformations
    abs_walltimes = np.array(pstats['walltimes']) + pstats['start_time']
    generation_times = np.diff(np.array([pstats['start_time']] + list(abs_walltimes)))
    parallel_fraction = np.array(pstats['workertimes'])/generation_times

    # NOTE: Possible x-axis are: generations, episodes, observations, walltimes

    for yscale in ['linear','log']:
        fig, ax = plt.subplots(figsize=figsize)
        pltUnp, = plt.plot(pstats[x], moving_average(pstats['return_unp']), label='parent ma')
        pltAvg, = plt.plot(pstats[x], moving_average(pstats['return_avg']), label='average ma')
        pltMax, = plt.plot(pstats[x], moving_average(pstats['return_max']), label='max ma')
        pltMin, = plt.plot(pstats[x], moving_average(pstats['return_min']), label='min ma')
        plt.gca().set_prop_cycle(None)
        pltUnpBack, = plt.plot(pstats[x], pstats['return_unp'], alpha=back_alpha, label='parent')
        pltAvgBack, = plt.plot(pstats[x], pstats['return_avg'], alpha=back_alpha, label='average')
        pltMaxBack, = plt.plot(pstats[x], pstats['return_max'], alpha=back_alpha, label='max')
        pltMinBack, = plt.plot(pstats[x], pstats['return_min'], alpha=back_alpha, label='min')
        ax.set_yscale(yscale)
        plt.ylabel('Return')
        plt.xlabel(x.capitalize())
        plt.legend(handles=[pltUnp, pltAvg, pltMax, pltMin, pltUnpBack, pltAvgBack, pltMaxBack, pltMinBack])
        fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew_' + yscale + '.pdf'), bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)
        pltUnpS, = plt.plot(pstats[x], moving_average(pstats['return_unp']), alpha=1, label='parent ma')
        plt.gca().set_prop_cycle(None)
        pltUnp, = plt.plot(pstats[x], pstats['return_unp'], alpha=back_alpha, label='parent raw')
        ax.set_yscale(yscale)
        plt.ylabel('Return')
        plt.xlabel(x.capitalize())
        plt.legend(handles=[pltUnpS, pltUnp])
        fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew_par_' + yscale + '.pdf'), bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)
        pltVarS, = plt.plot(pstats[x], moving_average(pstats['return_var']), label='ma')
        plt.gca().set_prop_cycle(None)
        pltVar, = plt.plot(pstats[x], pstats['return_var'], alpha=back_alpha, label='raw')
        ax.set_yscale(yscale)
        plt.ylabel('Return variance')
        plt.xlabel('Generations')
        plt.legend(handles=[pltVarS, pltVar])
        fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew_var_' + yscale + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    fig = plt.figure()
    pltRankS, = plt.plot(pstats[x], moving_average(pstats['unp_rank'], window=40), label='ma')
    plt.gca().set_prop_cycle(None)
    pltRank, = plt.plot(pstats[x], pstats['unp_rank'], alpha=back_alpha, label='raw')
    plt.ylabel('Unperturbed rank')
    plt.xlabel('Generations')
    plt.legend(handles=[pltRankS, pltRank])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_unprank.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], moving_average(generation_times), label='ma')
    plt.gca().set_prop_cycle(None)
    pltVar, = plt.plot(pstats[x], generation_times, alpha=back_alpha, label='raw')
    plt.ylabel('Walltime per generation')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_timeper.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], pstats['sigma'], label='sigma')
    plt.ylabel('Sigma')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_sigma.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], pstats['lr'], label='lr')
    plt.ylabel('Learning rate')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_lr.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], pstats['walltimes'], label='walltime')
    plt.ylabel('Walltime')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_time.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], moving_average(pstats['workertimes']), label='worker times')
    plt.ylabel('Worker time')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_worker_time.pdf'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats[x], moving_average(parallel_fraction), label='ma')
    plt.gca().set_prop_cycle(None)
    pltVar, = plt.plot(pstats[x], parallel_fraction, alpha=back_alpha, label='raw')
    plt.ylabel('Parallel fraction')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_parallel_fraction.pdf'), bbox_inches='tight')
    plt.close(fig)
