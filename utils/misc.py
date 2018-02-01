import inspect
import numpy as np


def get_equal_dicts(ds, ignored_keys=None):
    """Finds the dictionaries that are equal in a list of dictionaries.

    A group index value of g at index 0 of the output says that the 
    first dictionary belongs to group g. All dictionaries with equal group 
    index are equal.
    
    Args:
        ds (list): List of dictionaries to compare.
        ignored_keys (set, optional): Defaults to None. Keys not to include in the comparison.
    
    Returns:
        np.array: Array of group indices.
    """
    assert len(ds) >= 2, "The list must have at least two elements"
    match = False
    groups = np.zeros(len(ds), dtype=int)
    for (i, d) in enumerate(ds[1:]):
        i += 1
        for prev_i, prev_d in enumerate(ds[0:i]):
            if are_dicts_equal(d, prev_d, ignored_keys=ignored_keys):
                groups[i] = groups[prev_i]
                match = True
                break
        if not match:
            groups[i] = groups.max() + 1
    return groups


def are_dicts_equal(d1, d2, ignored_keys=None):
    """Test two dictionaries for equality while ignoring certain keys
    
    Args:
        d1 (dict): First dictionary
        d2 (dict): Second dictionary
        ignored_keys (set, optional): Defaults to None. A set of keys to ignore
    
    Returns:
        bool: Equality of the two dictionaries
    """
    for k1, v1 in d1.items():
        if (ignored_keys is None or k1 not in ignored_keys) and (k1 not in d2 or d2[k1] != v1):
            return False
    for k2, v2 in d2.items():
        if (ignored_keys is None or k2 not in ignored_keys) and (k2 not in d1):
            return False
    return True


def get_longest_sublists(l):
    """Return the longest sublist(s) in a list 
    
    Args:
        l (list): The list

    Returns:
        list: A list of the longest sublist(s)
    """
    length = length_of_longest(l)
    longest_list = list(filter(lambda l: len(l) == length, l))
    return longest_list


def length_of_longest(l):
    """Recursively find length of longest list in list of lists.
    
    Args:
        l (list): List of lists
    
    Returns:
        int: Length of longest sublist
    """
    if not isinstance(l, list):
        return 0
    return max([len(l),] + [len(subl) for subl in l if isinstance(subl, list)] + [length_of_longest(subl) for subl in l])


def get_inputs_from_dict(method, d):
    """
    Get a dictionary of the variables in the `NameSpace`, `d`, that match
    the `kwargs` of the given `method`.

    Useful for passing inputs from an `argparser` `NameSpace` to a method since
    standard behaviour would pass also any unknown keys from `d` to the
    `method`, resulting in error.
    """
    ins = inspect.getfullargspec(method)
    input_dict = {}
    for in_id, a in enumerate(ins.args):
        if a in d.keys():
            input_dict[a] = d[a]
    return input_dict