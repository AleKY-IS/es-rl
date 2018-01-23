import argparse
import os
import pickle
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


def monitor(args):
    # Get list of subdirectories (checkpoints)
    checkpoint_directories = [os.path.join(args.dir, di) for di in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, di))]

    # Monitoring loop
    while True:
        # IPython.embed()

        sl = load_statistics(checkpoint_directories)
        sl = invert_signs(sl)

        fig = plt.figure()
        handles = []
        for s in sl:
            handles.extend(plt.plot(s['generations'], moving_average(s['return_unp']), label="asd"))
        plt.ylabel('Return')
        plt.xlabel('Generations')
        plt.legend(handles=handles)
        fig.savefig(os.path.join(args.dir, 'monitoring' + '.pdf'))
        plt.close(fig)

        time.sleep(5)


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='Monitorer')
    parser.add_argument('--dir', type=str, metavar='DIR', help='The directory of checkpoints to monitor')
    args = parser.parse_args()

    try:
        monitor(args)
    except KeyboardInterrupt:
        print("Monitoring halted by user KeyboardInterrupt")

# python monitor.py --dir /Users/Jakob/mnt/Documents/es-rl/supervised-experiments/checkpoints/
