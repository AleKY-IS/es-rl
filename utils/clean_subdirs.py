"""Script that removes all files that equal given filenames in all sub directories of a directory.
"""

import os
import argparse
import IPython


if __name__ == '__main__':
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
        if i > 0:
            print('Removed {:d} {:s} in {:s}'.format(i, 'file' if i == 1 else 'files', root))
