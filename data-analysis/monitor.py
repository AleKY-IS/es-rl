import argparse
import os
import pickle
import sys
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas

from context import es
from es.utils import moving_average


def load_statistics(checkpoint_directories):
    statistics_list = []
    for chkpt_dir in checkpoint_directories:
        try:
            with open(os.path.join(chkpt_dir, 'stats.pkl'), 'rb') as filename:
                statistics_list.append(pickle.load(filename))
        except FileNotFoundError:
            pass

    return statistics_list


def invert_signs(sl):
    # Invert sign on negative returns (negative returns indicate a converted minimization problem)
    for s in sl:
        if (np.array(s['return_max']) < 0).all():
            for k in ['return_unp', 'return_avg', 'return_min', 'return_max']:
                s[k] = [-sk for sk in s[k]]
    return sl


def count_down(wait=60):
    for i in range(wait):
        print("Updating in {:s} seconds".format(str(wait-i)), end="\r")
        time.sleep(1)
    print("Updating plots and statistics...", end="\r")


def plot_mas(args, sl):
    fig = plt.figure()
    handles = []
    maxs = []
    for s in sl:
        ydata = moving_average(s['return_unp'])
        maxs.append(np.max(ydata))
        handles.extend(plt.plot(s['generations'], ydata, label="asd"))
    plt.ylabel('Return')
    plt.xlabel('Generations')
    plt.legend(handles=handles)
    plt.gca().set_ylim([0, np.median(maxs)*1.2])
    fig.savefig(os.path.join(args.dir, 'monitoring' + '.pdf'))
    plt.close(fig)


def print_stats(args, sl):
    maxs = []
    mins = []
    for s in sl:
        maxs.append(np.max(s['return_unp']))
        mins.append(np.min(s['return_unp']))
    print("=============== Statistics ===============")
    print("Max return {:>7.3f}".format(np.max(maxs)))
    print("Min return {:>7.3f}".format(np.min(mins)))
    print("==========================================")


def monitor(args):
    plt.ion()
    # Get list of subdirectories (checkpoints)
    checkpoint_directories = [os.path.join(args.dir, di) for di in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, di))]

    # Monitoring loop
    while True:
        # Load data
        sl = load_statistics(checkpoint_directories)
        sl = invert_signs(sl)
        # Print 
        print_stats(args, sl)
        # Plot
        plot_mas(args, sl)
        # Pause and count down until next update
        count_down(5)


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='Monitorer')
    parser.add_argument('--dir', type=str, metavar='DIR', help='The directory of checkpoints to monitor')
    args = parser.parse_args()

    try:
        monitor(args)
    except KeyboardInterrupt:
        print("\nMonitoring halted by user KeyboardInterrupt")

# python monitor.py --dir /Users/Jakob/mnt/Documents/es-rl/supervised-experiments/checkpoints/
