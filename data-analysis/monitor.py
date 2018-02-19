import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import IPython
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import context
from context import utils
import utils.plotting as plot
import utils.db as db
import utils.filesystem as fs
from utils.misc import get_equal_dicts, length_of_longest
from data_analysis import print_group_info, get_best, get_max_chkpt_int, invert_signs, get_checkpoint_directories


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
            old_mtimes = np.pad(old_mtimes, (0, len(mtimes)-len(old_mtimes)), mode='constant', constant_values=0)
        elif len(old_mtimes)-len(mtimes) > 0:
            mtimes = np.pad(mtimes, (0, len(old_mtimes)-len(mtimes)), mode='constant', constant_values=0)
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
        idxs = range(0, len(checkpoint_directories))
    # Strings and constants
    n_chars = len(str(n_files))
    f = '    {:' + str(n_chars) + 'd}/{:' + str(n_chars) + 'd} files loaded'
    s = ""
    # Loop over files to load (all or only changed ones)
    i_file = -1
    for i, chkpt_dir in zip(idxs, checkpoint_directories):
        try:
            algorithm_states_list[i] = torch.load(os.path.join(chkpt_dir, filename))
            i_file += 1
            if i_file + 1 != n_files:
                print(f.format(i_file + 1, n_files), end='\r')
        except Exception:
            s += "    Required files not (yet) present in: " + chkpt_dir + "\n"     

    # Remove any None
    algorithm_states_list = [s for s in algorithm_states_list if s is not None]
    print(f.format(i_file + 1, n_files), end='\n')
    if s:
        print(s[:-2])
    return algorithm_states_list


def sub_into_lists(algorithm_states, keys_to_monitor):
    for s in algorithm_states:
        for k in keys_to_monitor:
            if type(s['stats'][k][0]) is list:
                s['stats'][k] = [vals_group[0] for vals_group in s['stats'][k]]
                if k == 'lr' and 'lr' not in s.keys():
                    s['lr'] = s['stats'][k][0]


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

        plot.timeseries_final_distribution(list_of_series, label=k, ybins=len(list_of_series)*10)
        plt.savefig(os.path.join(args.monitor_dir, 'all-final-distribution-' + k + '.pdf'))
        plt.close()

        # Subset only those series that are done (or the one that is the longest)
        l = length_of_longest(list_of_series)
        indices = [i for i, series in enumerate(list_of_series) if len(series) == l]
        list_of_longest_series = [list_of_series[i] for i in indices]
        list_of_longest_genera = [list_of_genera[i] for i in indices]
        groups_longest_series = groups[indices]
        plot.timeseries_median_grouped(list_of_longest_genera, list_of_longest_series, groups_longest_series, xlabel='generations', ylabel=k)
        plt.savefig(os.path.join(args.monitor_dir, 'all-gen-' + k + '-series-mean-sd' + '.pdf'))
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

                plot.timeseries_final_distribution(list_of_series, label=k, ybins=len(list_of_series)*10)
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


def get_keys_to_monitor(algorithm_states):
    keys_to_monitor = set(algorithm_states[0]['stats']['do_monitor'])
    for s in algorithm_states[1:]:
        keys_to_monitor = keys_to_monitor.intersection(set(s['stats']['do_monitor']))
    return keys_to_monitor


def get_data(old_mtimes=None, old_data=None, timeout=30*60, checkevery=30):
    checkpoint_directories = get_checkpoint_directories(args.d)
    algorithm_states = load_data(checkpoint_directories, old_mtimes=old_mtimes, old_data=old_data)
    # Check if any data found
    if not algorithm_states:
        print("No data found.")
        print("Rechecking directory for files every " + str(checkevery) + " seconds for " + str(int(timeout/60)) + " minutes.")
        for i in range(0, timeout, checkevery):
            count_down(wait=checkevery, info_interval=1)
            checkpoint_directories = get_checkpoint_directories(args.d)
            algorithm_states = load_data(checkpoint_directories, old_mtimes=old_mtimes, old_data=old_data)
            if algorithm_states:
                return algorithm_states
            print("{:2.2f} minutes remaining".format((timeout - i-checkevery)/60))
        print("No data found to monitor after checking for " + str(int(timeout/60)) + " minutes.")
    return algorithm_states


def count_down(wait=60, count_down_started_at=None, info_interval=5):
    if count_down_started_at is not None:
        seconds_remaining = int(wait - (time.time() - count_down_started_at))
    else:
        seconds_remaining = wait
    for i in range(0, seconds_remaining, info_interval):
        print("Updating in {:s} seconds".format(str(seconds_remaining-i)), end="\r")
        time.sleep(info_interval)
    print("Updating...              ", end='\n')
    return time.time()


def monitor(args):
    this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
    # Get the root of the package locally and where monitored (may be the same)
    package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')

    # Get directory to monitor
    if not os.path.isabs(args.d):
        chkpt_dir = os.path.join(package_root_this_file, 'experiments', 'checkpoints')
        args.d = os.path.join(chkpt_dir, args.d)
    if not os.path.exists(args.d):
        os.mkdir(args.d)
    package_root_monitored_directory = fs.get_parent(args.d, 'es-rl')
    print("Monitoring: " + args.d)

    # Load data
    last_refresh = time.time()
    checkpoint_directories = get_checkpoint_directories(args.d)
    mtimes = fs.get_modified_times(checkpoint_directories, 'state-dict-algorithm.pkl')
    algorithm_states = get_data(timeout=args.t*60)
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
        package_parent_folder_monitored_directory = os.path.join(os.sep,*package_root_monitored_directory.split(os.sep)[:-1])
        # args.dbx_dir = os.sep + os.path.relpath(args.monitor_dir, package_parent_folder_monitored_directory)
        args.dbx_dir = os.sep + os.path.relpath(args.d, package_parent_folder_monitored_directory)
        token_file = os.path.join(this_file_dir_local, 'dropboxtoken.tok')
        assert os.path.exists(token_file)
        dbx = db.get_dropbox_client(token_file)

    ignored_keys = {'chkpt_dir', 'stats', 'sensitivities'}
    for s in algorithm_states:
        if s['optimize_sigma']:
            ignored_keys.add('sigma')
            break

    # Monitoring loop
    while True:
        # Prepare data
        print("Preparing data...")
        keys_to_monitor = get_keys_to_monitor(algorithm_states)
        invert_signs(algorithm_states, keys_to_monitor)
        sub_into_lists(algorithm_states, keys_to_monitor)
        # Find groups of algorithms
        groups = get_equal_dicts(algorithm_states, ignored_keys=ignored_keys)
        print_group_info(algorithm_states, groups, directory=args.monitor_dir)

        # Plot
        print("Creating and saving plots...")
        try:
            create_plots(args, algorithm_states, keys_to_monitor, groups)
        except:
            pass

        # Upload results to dropbox
        if args.c:
            # db.upload_directory(dbx, args.monitor_dir, args.dbx_dir)
            db.upload_directory(dbx, args.d, args.dbx_dir)

        # Break condition
        if wait_for_updates(args, last_refresh, get_max_chkpt_int(algorithm_states), mtimes):
            return
        # Load data
        print()
        last_refresh = time.time()
        algorithm_states = get_data(timeout=args.t*60, old_mtimes=mtimes, old_data=algorithm_states)
        checkpoint_directories = get_checkpoint_directories(args.d)
        mtimes = fs.get_modified_times(checkpoint_directories, 'state-dict-algorithm.pkl')


if __name__ == '__main__':
    # Parse inputs
    parser = argparse.ArgumentParser(description='Monitorer')
    parser.add_argument('-d', type=str, metavar='--directory', help='The directory of checkpoints to monitor.')
    parser.add_argument('-t', type=int, metavar='--timeout', default=30, help='If no files are modified during a period of timeout minutes, monitoring is stopped.')
    parser.add_argument('-c', action='store_true', help='Copying of monitor directory to dropbox.')
    parser.add_argument('-s', action='store_true', help='Silent mode.')
    args = parser.parse_args()

    if args.s:
        sys.stdout = open(os.devnull, 'w')

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