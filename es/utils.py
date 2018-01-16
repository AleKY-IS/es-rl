import inspect

import IPython
import matplotlib.pyplot as plt
import torch
import pprint
import os
import pickle
import pandas as pd

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
        s = "Gen {:3d} | Obs {:9d} | F {:5.2f} | Avg {:5.2f} | Max {:5.2f} | Min {:5.2f} | Var {:5.2f} | Rank {:3d} | Sig {:5.2f} | LR {:5.2f}".format(
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


def smooth(y, factor=10):
    if type(y) != list:
        y = list(y)
    return pd.Series(y).rolling(window=factor).mean()


def plot_stats(stats, chkpt_dir):
    # NOTE: Possible x-axis are: generations, episodes, observations, time
    fig = plt.figure()
    pltUnp, = plt.plot(stats['generations'], stats['reward_unp'], label='parent')
    pltAvg, = plt.plot(stats['generations'], stats['reward_avg'], label='average')
    pltMax, = plt.plot(stats['generations'], stats['reward_max'], label='max')
    pltMin, = plt.plot(stats['generations'], stats['reward_min'], label='min')
    plt.ylabel('rewards')
    plt.xlabel('generations')
    plt.legend(handles=[pltUnp, pltAvg, pltMax, pltMin])
    fig.savefig(chkpt_dir+'/rew_gen.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltUnp, = plt.plot(stats['generations'], stats['reward_unp'], label='parent')
    plt.ylabel('rewards')
    plt.xlabel('generations')
    plt.legend(handles=[pltUnp])
    fig.savefig(chkpt_dir+'/rew_gen_parent.pdf')
    plt.close(fig)

    fig = plt.figure()
    pltUnpS, = plt.plot(stats['generations'], smooth(stats['reward_unp']), alpha=1, label='smoothed parent')
    plt.gca().set_prop_cycle(None)
    pltUnp, = plt.plot(stats['generations'], stats['reward_unp'], alpha=.3, label='raw parent')
    plt.ylabel('rewards')
    plt.xlabel('generations')
    plt.legend(handles=[pltUnpS, pltUnp])
    fig.savefig(chkpt_dir+'/rew_gen_parent_smooth.pdf')
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
    plt.ylabel('walltimes')
    plt.xlabel('generations')
    plt.legend(handles=[pltVar])
    fig.savefig(chkpt_dir+'/time_gen.pdf')
    plt.close(fig)