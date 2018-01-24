import inspect
import os
import pickle
import pprint

import IPython
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch


def print_init(args, model, optimizer, lr_scheduler):
    """
    Print the initial message when training is started
    """
    s = "================= Evolver ==================\n"
    s += "Environment:          {:s}\n".format(args.env_name)
    s += "Workers:              {:d}\n".format(args.agents)
    s += "Generations:          {:d}\n".format(args.max_generations)
    s += "Sigma:                {:5.4f}\n".format(args.sigma)
    s += "Learning rate:        {:5.4f}\n".format(args.lr)
    s += "Batch size            {:<5d}\n".format(args.batch_size)
    s += "Optimizing sigma      {:s}\n".format(str(args.optimize_sigma))
    s += "Safe mutation         {:s}\n".format(str(args.safe_mutation))
    s += "Using CUDA            {:s}\n".format(str(args.cuda))
    s += "\n=================== Model ===================\n"
    s += pprint.pformat(model) + "\n"
    s += "\n================= Optimizer =================\n"
    s += str(type(optimizer)) + "\n"
    s += pprint.pformat(optimizer.state_dict()['param_groups']) + "\n"
    s += "\n================ LR scheduler ===============\n"
    s += str(type(lr_scheduler)) + "\n"
    s += "\n================== Running ==================\n"
    with open(os.path.join(args.chkpt_dir, 'init.log'), 'w') as f:
        f.write(s)
    print(s)


def print_iter(args, stats, workers_start_time, workers_end_time, loop_start_time):
    """
    Print information on a generation
    """
    print()
    lr = stats['lr'][-1] if type(stats['lr'][-1]) is not list else stats['lr'][-1][0]
    try:
        s = "Gen {:5d} | Obs {:9d} | F {:6.2f} | Avg {:6.2f} | Max {:6.2f} | Min {:6.2f} | Var {:7.2f} | Rank {:3d} | Sig {:5.4f} | LR {:5.4f}".format(
        stats['generations'][-1], stats['observations'][-1], stats['return_unp'][-1], stats['return_avg'][-1], stats['return_max'][-1], stats['return_min'][-1], stats['return_var'][-1], stats['unp_rank'][-1], stats['sigma'][-1], lr)
        print(s, end="")
    except Exception:
        print('In print_iter: Some number too large', end="")


def get_inputs_from_args(method, args):
    """
    Get a dictionary of the variables in the `NameSpace`, `args`, that match
    the `kwargs` of the given `method`.

    Useful for passing inputs from an `argparser` `NameSpace` to a method since
    standard behaviour would pass also any unknown keys from `args` to the
    `method`, resulting in error.
    """
    ins = inspect.getfullargspec(method)
    num_ins = len(ins.args)
    num_defaults = len(ins.defaults)
    num_required = num_ins - num_defaults
    input_dict = {}
    for in_id, a in enumerate(ins.args):
        if hasattr(args, a):
            input_dict[a] = getattr(args, a)
    return input_dict


def get_lr(optimizer):
    """
    Returns the current learning rate of an optimizer.
    If the model parameters are divided into groups, a list of 
    learning rates is returned. Otherwise, a single float is returned.
    """
    lr = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr.append(param_group['lr'])
    if len(lr) == 1:
        lr = lr[0]
    return lr


def load_checkpoint(restore_dir, file_path, model, optimizer, lr_scheduler, load_best=False):
    """
    Loads a checkpoint saved in the directory `restore_dir` which is a subfolder of the `file_path` of the
    calling function.
    """
    chkpt_dir = file_path+'/'+'/'.join([i for i in restore_dir.split('/') if i not in file_path.split('/')])
    try:
        if load_best:
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'best_model_state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'best_optimizer_state_dict.pth'))
        else:
            model_state_dict = torch.load(os.path.join(chkpt_dir, 'model_state_dict.pth'))
            optimizer_state_dict = torch.load(os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))
        with open(os.path.join(chkpt_dir, 'stats.pkl'), 'rb') as filename:
            stats = pickle.load(filename)
        lr_scheduler.last_epoch = stats['generations'][-1]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    except Exception:
        print("Checkpoint restore failed")
        raise Exception

    return chkpt_dir, model, optimizer, lr_scheduler, stats


def save_checkpoint(parent_model, optimizer, best_model_stdct, best_optimizer_stdct, stats, chkpt_dir):
    """
    Save a checkpoint of the `parent_model` and `optimizer` in the latest and best versions along with 
    statistics in the `stats` dictionary.
    """
    # Save latest model and optimizer state
    torch.save(parent_model.state_dict(), os.path.join(chkpt_dir, 'model_state_dict.pth'))
    torch.save(optimizer.state_dict(), os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))
    # torch.save(lr_scheduler.state_dict(), os.path.join(chkpt_dir, 'lr_scheduler_state_dict.pth'))
    # Save best model
    if best_model_stdct is not None:
        torch.save(best_model_stdct, os.path.join(chkpt_dir, 'best_model_state_dict.pth'))
        torch.save(best_optimizer_stdct, os.path.join(chkpt_dir, 'best_optimizer_state_dict.pth'))
    # Currently, learning rate scheduler has no state_dict and cannot be saved. It can be restored
    # by setting lr_scheduler.last_epoch = last generation index.
    with open(os.path.join(chkpt_dir, 'stats.pkl'), 'wb') as filename:
        pickle.dump(stats, filename, pickle.HIGHEST_PROTOCOL)


def moving_average(y, window=20, center=True):
    """
    Compute a moving average with of `window` observations in `y`. If `centered=True`, the 
    average is computed on `window/2` observations before and after the value of `y` in question. 
    If `centered=False`, the average is computed on the `window` previous observations.
    """
    if type(y) != list:
        y = list(y)
    return pd.Series(y).rolling(window=window, center=center).mean()


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
    figsize = (4, 3)
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

    # NOTE: Possible x-axis are: generations, episodes, observations, walltimes

    fig = plt.figure()
    pltUnp, = plt.plot(pstats[x], moving_average(pstats['return_unp']), label='parent ma')
    pltAvg, = plt.plot(pstats[x], moving_average(pstats['return_avg']), label='average ma')
    pltMax, = plt.plot(pstats[x], moving_average(pstats['return_max']), label='max ma')
    pltMin, = plt.plot(pstats[x], moving_average(pstats['return_min']), label='min ma')
    plt.gca().set_prop_cycle(None)
    pltUnpBack, = plt.plot(pstats[x], pstats['return_unp'], alpha=back_alpha, label='parent')
    pltAvgBack, = plt.plot(pstats[x], pstats['return_avg'], alpha=back_alpha, label='average')
    pltMaxBack, = plt.plot(pstats[x], pstats['return_max'], alpha=back_alpha, label='max')
    pltMinBack, = plt.plot(pstats[x], pstats['return_min'], alpha=back_alpha, label='min')
    plt.ylabel('Return')
    plt.xlabel(x.capitalize())
    plt.legend(handles=[pltUnp, pltAvg, pltMax, pltMin, pltUnpBack, pltAvgBack, pltMaxBack, pltMinBack])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew' + '.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltUnpS, = plt.plot(pstats[x], moving_average(pstats['return_unp']), alpha=1, label='parent ma')
    plt.gca().set_prop_cycle(None)
    pltUnp, = plt.plot(pstats[x], pstats['return_unp'], alpha=back_alpha, label='parent raw')
    plt.ylabel('Return')
    plt.xlabel(x.capitalize())
    plt.legend(handles=[pltUnpS, pltUnp])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew_par' + '.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltVarS, = plt.plot(pstats['generations'], moving_average(pstats['return_var']), label='ma')
    plt.gca().set_prop_cycle(None)
    pltVar, = plt.plot(pstats['generations'], pstats['return_var'], alpha=back_alpha, label='raw')
    plt.ylabel('Return variance')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVarS, pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_rew_var.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltRankS, = plt.plot(pstats['generations'], moving_average(pstats['unp_rank']), label='ma')
    plt.gca().set_prop_cycle(None)
    pltRank, = plt.plot(pstats['generations'], pstats['unp_rank'], alpha=back_alpha, label='raw')
    plt.ylabel('Unperturbed rank')
    plt.xlabel('Generations')
    plt.legend(handles=[pltRankS, pltRank])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_unprank.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats['generations'][:-1], moving_average(np.diff(pstats['walltimes'])), label='ma')
    plt.gca().set_prop_cycle(None)
    pltVar, = plt.plot(pstats['generations'][:-1], np.diff(pstats['walltimes']), alpha=back_alpha, label='raw')
    plt.ylabel('Walltime per generation')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_timeper.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats['generations'], pstats['sigma'], label='sigma')
    plt.ylabel('Sigma')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_sigma.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats['generations'], pstats['lr'], label='lr')
    plt.ylabel('Learning rate')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_lr.pdf'))
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(pstats['generations'], pstats['walltimes'], label='walltime')
    plt.ylabel('Walltime')
    plt.xlabel('Generations')
    plt.legend(handles=[pltVar])
    fig.savefig(os.path.join(chkpt_dir, x[0:3] + '_time.pdf'))
    plt.close(fig)


