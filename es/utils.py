import inspect
import IPython 


def get_inputs_from_args(method, args):
    # Get dict of inputs from args that match class __init__ method
    ins = inspect.getfullargspec(method)
    num_ins = len(ins.args)
    num_defaults = len(ins.defaults)
    num_required = num_ins - num_defaults
    input_dict = {}
    for in_id, a in enumerate(ins.args):
        if hasattr(args, a):
            input_dict[a] = getattr(args, a)

    return input_dict