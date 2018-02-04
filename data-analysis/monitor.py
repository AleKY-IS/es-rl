import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import IPython
import matplotlib.pyplot as plt
import numpy as np

import torch
import context
from context import utils
import utils.plotting as plot
import utils.db as db
import utils.filesystem as fs
from utils.misc import get_equal_dicts, length_of_longest


def load_data(checkpoint_directories, old_mtimes=None, old_data=None, best=False):
    # Parse inputs
    if best:
        filename = 'state-dict-best-algorithm.pkl'
    else:
        filename = 'state-dict-algorithm.pkl'
    n_tot_files = len(checkpoint_directories)
    if old_mtimes is not None:
        # Find files that have been modified
        assert old_data is not None, "If given modification times, must also get old data to overwrite"
        mtimes = fs.get_modified_times(checkpoint_directories, filename)
        if len(mtimes)-len(old_mtimes) > 0:
            mtimes = np.pad(old_mtimes, (None, len(mtimes)-len(old_mtimes)), 'constant')
        elif len(mtimes)-len(old_mtimes) < 0:
            mtimes = np.pad(mtimes, (None, len(old_mtimes)-len(mtimes)), 'constant')
        is_changed = ~np.equal(old_mtimes, mtimes)
        checkpoint_directories = [d for i, d in enumerate(checkpoint_directories) if is_changed[i]]
        n_files = len(checkpoint_directories)
        idxs = np.where(is_changed)[0]
        algorithm_states_list = old_data
        print("Loading " + str(n_files) + " modified files of " + str(n_tot_files) + " total files...")
    else:
        n_files = len(checkpoint_directories)
        print("Loading " + str(n_files) + " files...")
        algorithm_states_list = [None]*len(checkpoint_directories)
        idxs = range(0,len(checkpoint_directories))
    # Strings and constants
    n_chars = len(str(n_files))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} files loaded'
    s = ""
    print(f.format(1, len(checkpoint_directories)), end='\r')
    # Loop over files to load (all or only changed ones)
    i_file = 0
    for i, chkpt_dir in zip(idxs, checkpoint_directories):
        try:
            algorithm_states_list[i] = (torch.load(os.path.join(chkpt_dir, filename)))
        except Exception:
            s += "    Required files not (yet) present in: " + chkpt_dir + "\n"
        if i_file + 1 == n_files:
            print(f.format(i_file + 1, n_files), end='\n')
        else:
            print(f.format(i_file + 1, n_files), end='\r')
        i_file += 1
    if s:
        print(s[:-2])
    return algorithm_states_list


def get_max_chkpt_int(algorithm_states):
    """Get the maximum time in seconds between checkpoints.
    """
    max_chkpt_int = -1
    for s in algorithm_states:
        max_chkpt_int = max(s['chkpt_int'], max_chkpt_int)
    return max_chkpt_int


def invert_signs(algorithm_states, keys_to_monitor):
    """Invert sign on negative returns.
    
    Negative returns indicate a converted minimization problem so this converts the problem 
    considered to maximization which is the standard in the algorithms.

    Args:
        algorithm_states ([type]): [description]
        keys_to_monitor ([type]): [description]
    """
    if keys_to_monitor == 'all':
        keys_to_monitor = {'return_unp', 'return_max', 'return_min', 'return_avg'}
    for s in algorithm_states:
        if (np.array(s['stats']['return_max']) < 0).all():
            for k in {'return_unp', 'return_max', 'return_min', 'return_avg'}.intersection(keys_to_monitor):
                s['stats'][k] = [-retrn for retrn in s['stats'][k]]


def sub_into_lists(algorithm_states, keys_to_monitor):
    for s in algorithm_states:
        for k in keys_to_monitor:
            if type(s['stats'][k][0]) is list:
                s['stats'][k] = [vals_group[0] for vals_group in s['stats'][k]]


def get_keys_to_monitor(algorithm_states):
    keys_to_monitor = set(algorithm_states[0]['stats']['do_monitor'])
    for s in algorithm_states[1:]:
        keys_to_monitor = keys_to_monitor.intersection(set(s['stats']['do_monitor']))
    return keys_to_monitor


def create_plots(args, algorithm_states, keys_to_monitor, groups):
    unique_groups = set(groups)
    n_keys = len(keys_to_monitor)
    n_chars = len(str(n_keys))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} monitored keys plotted'
    for i_key, k in enumerate(keys_to_monitor):
        list_of_series = [s['stats'][k] for s in algorithm_states]
        list_of_genera = [s['stats']['generations'] for s in algorithm_states]

        plot.timeseries(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-gen-' + k + "-series" + '.pdf'))
        plt.close()

        plot.timeseries_distribution(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-gen-' + k + "-distribution" + '.pdf'))
        plt.close()

        plot.timeseries_median(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-gen-' + k + "-median" + '.pdf'))
        plt.close()

        plot.timeseries_final_distribution(list_of_series, label=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-final-distribution-' + k + '.pdf'))
        plt.close()

        # Subset only those series that are done (or the one that is the longest)
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        list_of_longest_series = [list_of_series[i] for i in indices]
        list_of_longest_genera = [list_of_genera[i] for i in indices]
        groups_longest_series = groups[indices]
        plot.timeseries_median_grouped(list_of_longest_genera, list_of_longest_series, groups_longest_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-gen-' + k + '-series-grouped' + '.pdf'))
        plt.close()

        if len(unique_groups) > 1:
            for g in unique_groups:
                gstr = '{0:02d}'.format(g)
                g_indices = np.where(groups == g)[0]
                group_alg_states = [algorithm_states[i] for i in g_indices]

                list_of_series = [s['stats'][k] for s in group_alg_states]
                list_of_genera = [s['stats']['generations'] for s in group_alg_states]

                plot.timeseries(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                plt.savefig(os.path.join(args.monitor_dir, 'group-' + gstr + '-gen-' + k + "-series" + '.pdf'))
                plt.close()

                plot.timeseries_distribution(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                plt.savefig(os.path.join(args.monitor_dir, 'group-' + gstr + '-gen-' + k + "-distribution" + '.pdf'))
                plt.close()

                plot.timeseries_median(list_of_genera, list_of_series, xlabel='generations', ylabel=k)
                plt.savefig(os.path.join(args.monitor_dir, 'group-' + gstr + '-gen-' + k + "-median" + '.pdf'))
                plt.close()

                plot.timeseries_final_distribution(list_of_series, label=k)
                plt.savefig(os.path.join(args.monitor_dir, 'group-' + gstr + '-final-distribution-' + k + '.pdf'))
                plt.close()

        if i_key + 1 == n_keys:
            print(f.format(i_key+1, n_keys), end='\n')
        else:
            print(f.format(i_key+1, n_keys), end='\r')


def wait_for_updates(args, last_refresh, max_chkpt_int, mtimes_last):
    """Wait for updates to the chyeckpoint directories.

    If no updates are seen after waiting more than the maximum checkpoint
    interval, returns False. Otherwise returns True.
    """
    print("Waiting 'max checkpoint interval' + 10% = " + str(int(max_chkpt_int*1.1)) + " seconds before checking for updates...")
    count_down(count_down_started_at=last_refresh, wait=max_chkpt_int*1.1)
    checkpoint_directories = get_checkpoint_directories(args.d)
    mtimes = fs.get_modified_times(checkpoint_directories, 'state-dict-algorithm.pkl')
    if mtimes == mtimes_last:
        print("Monitoring stopped since loaded data did not change for " + str(int(max_chkpt_int*1.1)) + " seconds.")
        return True
    return False


def get_data(checkpoint_directories, old_mtimes=None, old_data=None, timeout=30*60, checkevery=30):
    algorithm_states = load_data(checkpoint_directories, old_mtimes=old_mtimes, old_data=old_data)
    # Check if any data found
    if not algorithm_states:
        print("No data found.")
        print("Rechecking directory for files every " + str(checkevery) + " seconds for " + str(int(timeout/60)) + "    minutes.")
        for i in range(0, timeout, checkevery):
            count_down(wait=checkevery)
            algorithm_states = load_data(checkpoint_directories, old_mtimes=old_mtimes, old_data=old_data)
            if algorithm_states:
                return algorithm_states
        print("No data found to monitor after checking for " + str(int(timeout/60)) + " minutes.")
    else:
        return algorithm_states


def count_down(wait=60, count_down_started_at=time.time(), info_interval=5):
    seconds_remaining = wait - (time.time() - count_down_started_at)
    seconds_remaining = int(seconds_remaining) if seconds_remaining > 0 else 0
    if info_interval == -1:
        info_interval = seconds_remaining
    for i in range(0, seconds_remaining, info_interval):
        print("Updating in {:s} seconds".format(str(seconds_remaining-i)), end="\r")
        time.sleep(info_interval)
    print("Updating...              ", end='\n')
    return time.time()


def get_checkpoint_directories(dir):
    return [os.path.join(args.d, di) for di in os.listdir(args.d) if os.path.isdir(os.path.join(args.d, di)) and di != 'monitoring']


def monitor(args):
    this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
    # Get the root of the package locally and where monitored (may be the same)
    package_root_monitored = fs.get_parent(args.d, 'es-rl')

    # Get directory to monitor
    if not os.path.isabs(args.d):
        chkpt_dir = os.path.join(package_root_monitored, 'experiments', 'checkpoints')
        args.d = os.path.join(chkpt_dir, args.d)
    if not os.path.exists(args.d):
        os.mkdir(args.d)

    # Load data
    last_refresh = time.time()
    checkpoint_directories = get_checkpoint_directories(args.d)
    mtimes = fs.get_modified_times(checkpoint_directories, 'state-dict-algorithm.pkl')
    algorithm_states = get_data(checkpoint_directories, timeout=args.t*60)
    if not algorithm_states:
        print("Monitoring stopped. No data available after " + str(args.t) + " minutes.")
        return

    # Create directory for monitoring plots
    monitor_dir = os.path.join(args.d, 'monitoring')
    if not os.path.exists(monitor_dir):
        os.mkdir(monitor_dir)
    args.monitor_dir = monitor_dir

    # Setup drobbox
    if args.c:
        package_root_parent = os.path.join(os.sep,*package_root_monitored.split(os.sep)[:-1])
        args.dbx_dir = os.sep + os.path.relpath(args.monitor_dir, package_root_parent)
        dbx = db.get_dropbox_client()

    # Monitoring loop
    while True:
        # Prepare data
        print("Preparing data...")
        keys_to_monitor = get_keys_to_monitor(algorithm_states)
        invert_signs(algorithm_states, keys_to_monitor)
        sub_into_lists(algorithm_states, keys_to_monitor)
        # Find groups of algorithms
        groups = get_equal_dicts(algorithm_states, ignored_keys={'chkpt_dir', 'stats'})

        # Plot
        print("Creating and saving plots...")
        create_plots(args, algorithm_states, keys_to_monitor, groups)

        # Upload results to dropbox
        if args.c:
            db.copy_dir(dbx, args.monitor_dir, args.dbx_dir)

        # Break condition
        if wait_for_updates(args, last_refresh, get_max_chkpt_int(algorithm_states), mtimes):
            return
        # Load data
        print()
        last_refresh = time.time()
        algorithm_states = get_data(checkpoint_directories, timeout=args.t*60, old_mtimes=mtimes, old_data=algorithm_states)
        mtimes = fs.get_modified_times(checkpoint_directories, 'state-dict-algorithm.pkl')


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='Monitorer')
    parser.add_argument('-d', type=str, metavar='--directory', help='The directory of checkpoints to monitor.')
    parser.add_argument('-t', type=int, metavar='--timeout', default=30, help='If no files are modified during a period of timeout minutes, monitoring is stopped.')
    parser.add_argument('-c', action='store_true', help='Copying of monitor directory to dropbox.')
    args = parser.parse_args()

    # Colormap
    # plt.rcParams['image.cmap'] = 'magma'
    # plt.rcParams['image.cmap'] = 'inferno'
    # plt.rcParams['image.cmap'] = 'plasma'
    plt.rcParams['image.cmap'] = 'viridis'

    try:
        monitor(args)
    except KeyboardInterrupt:
        print("\nMonitoring halted by user KeyboardInterrupt")

"""
SSHFS
sshfs s132315@login.hpc.dtu.dk:/zhome/c2/b/86488 ~/mnt

LINUX
python monitor.py -d ~/mnt/Documents/es-rl/experiments/checkpoints/E001-SM/ -c

MAC
python monitor.py -d /Users/Jakob/mnt/Documents/es-rl/experiments/checkpoints/ES001-SM/ -c

python monitor.py -d sftp://s132315@login.hpc.dtu.dk/zhome/c2/b/86488/Documents/es-rl/experiments/checkpoints/E001-SM
"""