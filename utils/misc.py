import inspect
import itertools

import IPython
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
    groups = np.zeros(len(ds), dtype=int)
    match = False
    for i, d in enumerate(ds[1:]):
        i += 1
        for prev_i, prev_d in enumerate(ds[:i]):
            if are_dicts_equal(d, prev_d, ignored_keys=ignored_keys):
                groups[i] = groups[prev_i]
                match = True
                break
        # If no matches, create new group
        if not match:
            groups[i] = groups.max() + 1
        match = False
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
    return max([len(l)] + [len(subl) for subl in l if isinstance(subl, list)] + [length_of_longest(subl) for subl in l])


def get_inputs_from_dict_class(c, d, recursive=False):
    """
    The same as `get_inputs_from_dict` but for classes that may have parenting classes.

    If `recursive` is `True`, the function gets the input of the `__init__` method of 
    the class and every parenting class which is in `d`.
    """
    if not recursive:
        input_dict = get_inputs_from_dict(c.__init__, d)
    else:
        input_dict = {}
        if type(c) is not tuple:
            c = (c,)
        # Get inputs for each given class
        for ci in c:
            input_dict = {**input_dict, **get_inputs_from_dict(ci.__init__, d)}
            # Call self on each of the classes' parenting classes
            for parent_c in ci.__bases__:
                input_dict = {**input_dict, **get_inputs_from_dict(parent_c.__init__, d)}
    return input_dict
        


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


def is_nearly_equal(a, b, eps=None):
    """Compares floating point numbers
    
    Args:
        a (float): First number
        b (float): Second number
        eps (float, optional): Defaults to None. Absolute cutoff value for near equality
    
    Raises:
        NotImplementedError: [description]
    
    Returns:
        bool: Whether a and b are nearly equal or not
    """
    # Validate input
    assert(not hasattr(a, "__len__") and not hasattr(b, "__len__"), 
           'Inputs must be scalar')
    # Use machine precision if none given
    if eps is None:
        assert type(a) == type(b), "Cannot infer machine precision if types are different"
        # Numpy type
        if type(a).__module__ == np.__name__:
            finfo = np.finfo(type(a))
        else: 
            raise NotImplementedError('Cannot infer machine epsilon for non-numpy types')
    # Compare
    diff = np.abs(a - b)
    if a == b:
        # Shortcut and handles infinities
        return True
    elif (a == 0 or b == 0 or diff < finfo.min):
        # a or b or both are zero or are very close to it.
        # Relative error is less meaningful here, so we check absolute error.
        return diff < (finfo.eps * finfo.min)
    else:
        # Relative error
        return diff / np.min(np.abs(a) + np.abs(b), finfo.max) < finfo.eps


def list_of_dicts_to_dict_of_lists(ld):
    return dict(zip(ld[0],zip(*[d.values() for d in ld])))


def dict_of_lists_to_list_of_dicts(dl):
    return [dict(zip(dl,t)) for t in zip(*dl.values())]


""" 
public static boolean nearlyEqual(float a, float b, float epsilon) {
final float absA = Math.abs(a);
final float absB = Math.abs(b);
final float diff = Math.abs(a - b);

if (a == b) { // shortcut, handles infinities
    return true;
} else if (a == 0 || b == 0 || diff < Float.MIN_NORMAL) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    return diff < (epsilon * Float.MIN_NORMAL);
} else { // use relative error
    return diff / Math.min((absA + absB), Float.MAX_VALUE) < epsilon;
}
}
"""