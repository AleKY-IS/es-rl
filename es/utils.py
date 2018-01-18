import inspect
import os
import pickle
import pprint

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch


def print_init(args, model, optimizer, lr_scheduler):
    print("================= Evolver ==================")
    print("Environment:          {:s}".format(args.env_name))
    print("Workers:              {:d}".format(args.agents))
    print("Generations:          {:d}".format(args.max_generations))
    print("Sigma:                {:5.4f}".format(args.sigma))
    print("Learning rate:        {:5.4f}".format(args.lr))
    print("Restore point:        {:s}".format(args.restore))
    print("\n=================== Model ===================")
    print(model)
    print("\n================= Optimizer =================")
    pprint.pprint(optimizer.state_dict()['param_groups'])
    print("\n================ LR scheduler ===============")
    print(lr_scheduler)
    print()
    print("\n================== Running ==================")


def print_iter(args, stats, workers_start_time, workers_end_time, loop_start_time):
    print()
    try:
        s = "Gen {:5d} | Obs {:9d} | F {:6.2f} | Avg {:6.2f} | Max {:6.2f} | Min {:6.2f} | Var {:7.2f} | Rank {:3d} | Sig {:5.4f} | LR {:5.4f}".format(
        stats['generations'][-1], stats['observations'][-1], stats['reward_unp'][-1], stats['reward_avg'][-1], stats['reward_max'][-1], stats['reward_min'][-1], stats['reward_var'][-1], stats['unp_rank'][-1], stats['sigma'][-1], stats['lr'][-1])
        print(s, end="")
    except Exception:
        print('Some number too large', end="")


def get_inputs_from_args(method, args):
    """
    Get dict of inputs from args that match class __init__ method
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
    except Exception:
        print("Checkpoint restore failed")
        raise Exception
    lr_scheduler.last_epoch = stats['generations'][-1]
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    return chkpt_dir, model, optimizer, lr_scheduler, stats


def save_checkpoint(parent_model, optimizer, best_model_stdct, best_optimizer_stdct, stats, chkpt_dir):
    # Save latest model and optimizer state
    torch.save(parent_model.state_dict(), os.path.join(chkpt_dir, 'model_state_dict.pth'))
    torch.save(optimizer.state_dict(), os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))
    # torch.save(lr_scheduler.state_dict(), os.path.join(chkpt_dir, 'lr_scheduler_state_dict.pth'))
    # Save best model
    torch.save(best_model_stdct, os.path.join(chkpt_dir, 'best_model_state_dict.pth'))
    torch.save(best_optimizer_stdct, os.path.join(chkpt_dir, 'best_optimizer_state_dict.pth'))
    # Currently, learning rate scheduler has no state_dict and cannot be saved. It can be restored
    # by setting lr_scheduler.last_epoch = last generation index.
    with open(os.path.join(chkpt_dir, 'stats.pkl'), 'wb') as filename:
        pickle.dump(stats, filename, pickle.HIGHEST_PROTOCOL)


def moving_average(y, window=20, center=True):
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
    """

    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    figsize = (4, 3)
    # NOTE: Possible x-axis are: generations, episodes, observations, walltimes
    for x in ['generations', 'observations', 'walltimes']:
        fig = plt.figure(figsize=figsize)
        pltUnp, = plt.plot(stats[x], stats['reward_unp'], label='parent')
        pltAvg, = plt.plot(stats[x], stats['reward_avg'], label='average')
        pltMax, = plt.plot(stats[x], stats['reward_max'], label='max')
        pltMin, = plt.plot(stats[x], stats['reward_min'], label='min')
        plt.ylabel('rewards')
        plt.xlabel(x)
        plt.legend(handles=[pltUnp, pltAvg, pltMax, pltMin])
        fig.savefig(chkpt_dir+'/rew_' + x[0:3] + '.pdf')
        plt.close(fig)

        fig = plt.figure()
        pltUnp, = plt.plot(stats[x], stats['reward_unp'], label='parent')
        plt.ylabel('rewards')
        plt.xlabel(x)
        plt.legend(handles=[pltUnp])
        fig.savefig(chkpt_dir+'/rewpar_' + x[0:3] + '.pdf')
        plt.close(fig)

        fig = plt.figure()
        pltUnp, = plt.plot(stats[x], moving_average(stats['reward_unp']), label='smoothed parent')
        pltAvg, = plt.plot(stats[x], moving_average(stats['reward_avg']), label='smoothed average')
        pltMax, = plt.plot(stats[x], moving_average(stats['reward_max']), label='smoothed max')
        pltMin, = plt.plot(stats[x], moving_average(stats['reward_min']), label='smoothed min')
        plt.gca().set_prop_cycle(None)
        pltUnpBack, = plt.plot(stats[x], stats['reward_unp'], alpha=0.3, label='parent')
        pltAvgBack, = plt.plot(stats[x], stats['reward_avg'], alpha=0.3, label='average')
        pltMaxBack, = plt.plot(stats[x], stats['reward_max'], alpha=0.3, label='max')
        pltMinBack, = plt.plot(stats[x], stats['reward_min'], alpha=0.3, label='min')
        plt.ylabel('rewards')
        plt.xlabel(x)
        plt.legend(handles=[pltUnp, pltAvg, pltMax, pltMin, pltUnpBack, pltAvgBack, pltMaxBack, pltMinBack])
        fig.savefig(chkpt_dir+'/rewsmo_' + x[0:3] + '.pdf')
        plt.close(fig)

        #IPython.embed()
        fig = plt.figure()
        pltUnpS, = plt.plot(stats[x], moving_average(stats['reward_unp']), alpha=1, label='smoothed parent')
        plt.gca().set_prop_cycle(None)
        pltUnp, = plt.plot(stats[x], stats['reward_unp'], alpha=.3, label='raw parent')
        plt.ylabel('rewards')
        plt.xlabel(x)
        plt.legend(handles=[pltUnpS, pltUnp])
        fig.savefig(chkpt_dir+'/rewparsmo_' + x[0:3] + '.pdf')
        plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'], stats['reward_var'], label='reward variance')
    plt.ylabel('reward variance')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/rewvar_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'], stats['unp_rank'], label='unperturbed rank')
    plt.ylabel('unperturbed rank')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/unprank_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'], stats['sigma'], label='sigma')
    plt.ylabel('sigma')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/sigma_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'], stats['lr'], label='lr')
    plt.ylabel('learning rate')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/lr_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'], stats['walltimes'], label='lr')
    plt.ylabel('walltime')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/time_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltVar, = plt.plot(stats['generations'][:-1], np.diff(stats['walltimes']), label='lr')
    plt.ylabel('walltime per generation')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/timeper_gen.pdf')
    plt.close(fig)
