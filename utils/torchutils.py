import math
from collections import OrderedDict

import IPython
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def get_names_dict(model):
    """
    Recursive walk to get names including path
    """
    names = {}
    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name]=module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)
    _get_names(model)
    return names


def torch_summarize(model, input_size, return_meta=False):
    """
    Summarizes torch model by showing trainable parameters and weights.
    
    author: wassname
    url: https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7
    license: MIT
    
    Modified from:
    - https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
    - https://gist.github.com/wassname/0fb8f95e4272e6bdd27bd7df386716b7/
    
    Usage:
        import torchvision.models as models
        model = models.alexnet()
        df = torch_summarize(model=model, input_size=(3, 224, 224))
        print(df)
        
                     name class_name        input_shape       output_shape  n_parameters
        1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
        2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
        ...
    """

    def register_hook(module):
        # Define hook
        def hook(module, input, output):
            name = ''
            for key, item in names.items():
                if item == module:
                    name = key
            # Get class name and set module index
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = module_idx + 1
            # Prepare summary entry for this module
            summary[m_key] = OrderedDict()
            summary[m_key]['name'] = name
            summary[m_key]['class_name'] = class_name
            # Input and output shape
            summary[m_key]['input_shape'] = (-1, ) + tuple(input[0].size())[1:]
            summary[m_key]['output_shape'] = (-1, ) + tuple(output.size())[1:]
            # Weight dimensions
            summary[m_key]['weight_shapes'] = list([tuple(p.size()) for p in module.parameters()])
            # Number of parameters in layers
            summary[m_key]['n_parameters'] = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])            
            summary[m_key]['n_trainable'] = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])

        # Append 
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Put model in evaluation mode (required for e.g. batchnorm layers)
    model.eval()
    # Names are stored in parent and path+name is unique not the name
    names = get_names_dict(model)
    # Check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))
    # Move parameters to CUDA if relevant
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    # Create properties
    summary = OrderedDict()
    hooks = []
    # Register hook on all modules of model
    model.apply(register_hook)
    # Make a forward pass to evaluate registered hook functions
    # and build summary
    model(x)
    # Remove all the registered hooks from the model again and
    # return it in the state it was given.
    for h in hooks:
        h.remove()
    # Make dataframe
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    # Create additional info
    if return_meta:
        meta = {'total_parameters': df_summary.n_parameters.sum(),
                'total_trainable': df_summary.n_trainable.sum(),
                'layers': df_summary.shape[0],
                'trainable_layers': (df_summary.n_trainable != 0).sum()}
        df_meta = pd.DataFrame.from_dict(meta, orient='index')
        return df_summary, df_meta
    else:
        return df_summary


def get_mean_and_std(dataset, dim=0):
    """Compute the mean and standard deviation of a torchvision dataset

    Usage:
        from torchvision import datasets, transforms
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        mean, std = get_mean_and_std(dataset)
        dataset = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (mean,), (std,))
                                ]))

        from torchvision import datasets, transforms
        dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
        mean, std = get_mean_and_std(dataset)
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (mean,), (std,))
                                ]))

    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    IPython.embed()
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inp, tgt in dataloader:
        for i in range(1):
            mean[i] += inp[:,i,:,:].mean()
            std[i] += inp[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def calculate_xavier_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function. The values are as follows:
    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn` name, i.e. module name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = calculate_xavier_gain(nn.modules.ReLU)
    """
    linear_fns = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d , nn.ConvTranspose2d , nn.ConvTranspose3d]
    islinear = any([nonlinearity.__name__ == lf.__name__ for lf in linear_fns])
    if islinear or nonlinearity.__name__ == nn.Sigmoid.__name__:
        return 1
    elif nonlinearity.__name__ == nn.Tanh.__name__:
        return 5.0 / 3
    elif nonlinearity.__name__ == nn.ReLU.__name__:
        return math.sqrt(2.0)
    elif nonlinearity.__name__ == nn.LeakyReLU.__name__:
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


if __name__ == '__main__':
    # Test on alexnet
    import torchvision.models as models
    model = models.alexnet()
    df = torch_summarize(input_size=(3, 224, 224), model=model)
    print(df)
