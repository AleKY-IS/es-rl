import inspect
import os

import IPython
import numpy as np


def get_modified_times(dirs, f):
    """Return the times at which the `file` in the directories `dirs` was modified.
    
    Args:
        dirs (str): Path
        file (str): File name
    
    Returns:
        list: List of modification times
    """
    mtimes = []
    for d in dirs:
        path = os.path.join(d, f)
        if os.path.exists(path):
            mtimes.append(os.path.getmtime(path))
        else:
            mtimes.append(0)
    return mtimes


def get_parent(directory, parent_folder_name):
    """Get the absolute path of some parenting folder of a directory.
    
    Args:
        directory (str): The path to search.
        parent_folder_name (str): Name of the parenting folder.
    
    Returns:
        str, None: Directory of parenting folder. None of not found on path.
    """
    if parent_folder_name not in directory.split(os.sep):
        return None
    parent_directory = ['/']
    for d in directory.split(os.sep):
        if d != parent_folder_name:
            parent_directory.append(d)
        if d == parent_folder_name:
            parent_directory.append(d)
            break
    return os.path.join(*parent_directory)


def longest_common_suffix(list_of_paths):
    reversed_paths = [os.path.join(*s.split(os.sep)[::-1]) for s in list_of_paths]
    reversed_lcs = os.path.commonprefix(reversed_paths)
    lcs = os.path.join(*reversed_lcs.split(os.sep)[::-1])
    if lcs:
        return os.path.join(os.sep, lcs)
    else:
        return None