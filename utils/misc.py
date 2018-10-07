import inspect
import itertools
import collections
import sys

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
    groups = np.zeros(len(ds), dtype=int)
    if len(ds) == 1:
        return groups
    match = False
    for i, d in enumerate(ds[1:]):
        i += 1
        for prev_i, prev_d in enumerate(ds[:i]):
            is_equal = are_dicts_equal(d, prev_d, ignored_keys=ignored_keys)
            if is_equal:
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
        try:
            if (ignored_keys is None or k1 not in ignored_keys) and (k1 not in d2 or d2[k1] != v1):
                return False
        except RuntimeError as e:
            raise type(e)(str(e) + '. The key was k1={}'.format(k1)).with_traceback(sys.exc_info()[2])
    for k2, v2 in d2.items():
        try:
            if (ignored_keys is None or k2 not in ignored_keys) and (k2 not in d1):
                return False
        except RuntimeError as e:
            raise type(e)(str(e) + '. The key was k2={}'.format(k1)).with_traceback(sys.exc_info()[2])
        # if (ignored_keys is None or k2 not in ignored_keys) and (k2 not in d1):
            # return False
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
    lengths = [len(l_i) for l_i in l]
    return max(lengths)


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
                input_dict = {**input_dict, **get_inputs_from_dict_class(parent_c, d, recursive=True)}
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


def isfloat(x):
    """Checks if x is convertible to float type.
    If also checking if x is convertible to int type, this must be done before since
    this implies x is also convertible to float.
    """
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    """Checks if x is convertible to int type.
    If x is convertible to int type it is also convertible to float type.
    """
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def to_numeric(v):
    if isint(v):
        return int(float(v))
    elif isfloat(v):
        return float(v)
    else:
        return v


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


def get_indices_of_A_in_B(A, B):
    """Return the set of indices into B of the elements in A that occur in B
    
    Parameters
    ----------
    A : list
        The "needles"
    B : list
        The "haystack"
    Returns
    -------
    list
        Indices into B of elements in A occuring in B
    """
    s = set(B)
    return [i for i, e in enumerate(A) if e in s]


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: itertools.chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    collections.deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
