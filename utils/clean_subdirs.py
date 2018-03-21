"""Script that removes all files that equal given filenames in all sub directories of a directory.
"""

import os
import argparse
import IPython


if __name__ == '__main__':
    # Ask to continue
    r = input('This script should not be run while algorithms are executing as this risks deleting their checkpoints.\nProceed? (y/n) ')
    if r not in ['y', 'Y']:
        print('Script ended. No files deleted.')
        exit(0)
    IPython.embed()
    # Input
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('-d', type=str, metavar='directory', help='Directory to clean')
    parser.add_argument('-f', type=str, nargs='+', metavar='files', help='File(s) to remove')
    args = parser.parse_args()

    # Validate
    if args.d is None and args.f is None:
        args.d = '/home/jakob/Dropbox/es-rl/experiments/checkpoints'
        args.f = ['state-dict-algorithm.pkl', 'state-dict-optimizer.pkl', 'state-dict-model.pkl']
    assert args.d is not None and args.f is not None

    # Run
    for root, directories, filenames in os.walk(args.d):
        i = 0
        for filename in filenames:
            if filename in args.f:
                os.remove(os.path.join(root, filename))
                i += 1
        if len(filenames) == 1 and filenames[0] == 'init.log':
            os.remove(os.path.join(root, filenames[0]))
            os.rmdir(os.path.join(root))
            i += 1
        if i > 0:
            print('Removed {:d} {:s} in {:s}'.format(i, 'file' if i == 1 else 'files', root))
