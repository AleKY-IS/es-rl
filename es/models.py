from functools import partial
from itertools import chain, islice, tee

import gym
import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import _addindent

from context import utils
from utils.torchutils import summarize_model, calculate_xavier_gain


def model_weight_initializer(model):
    def current_and_next(some_iterable):
        items, nexts = tee(some_iterable)
        nexts = chain(islice(nexts, 1, None), [None])
        return zip(items, nexts)

    def previous_and_next(some_iterable):
        prevs, items, nexts = tee(some_iterable, 3)
        prevs = chain([None], prevs)
        nexts = chain(islice(nexts, 1, None), [None])
        return zip(prevs, items, nexts)
    
    for module, next_module in current_and_next(model.modules()):
        IPython.embed()
        try:
            gain = nn.init.calculate_gain(next_module)
        except:
            continue
        if isinstance(module, nn.Conv2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            module.weight.data.normal_(0, np.sqrt(2. / n))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            module.bias.data.zero_()
        

def capsule_softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class IdentityFunction(nn.Module):
    """Identity activation function module for use when no activation is needed but a function call is.
    """
    def __init__(self):
        super(IdentityFunction, self).__init__()

    def forward(self, x):
        return x


class AbstractESModel(nn.Module):
    """Abstract models class for models that are trained by evolutionary methods.
    It has methods for counting parameters, layers and tensors.
    """
    @property
    def summary(self):
        if not hasattr(self, '_summary'):
            self._summary = summarize_model(self, self.in_dim)
        return self._summary

    def count_parameters(self, only_trainable=True):
        """Return the number of [trainable] parameters in this model.
        """
        return self._count_parameters(self, only_trainable=only_trainable)

    @staticmethod
    def _count_parameters(m, only_trainable=True):
        """Count the number of [trainable] parameters in a pytorch model.
        """
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return int(m.summary[k].sum())

    def count_tensors(self, only_trainable=True):
        return self._count_tensors(self, only_trainable=only_trainable)

    @staticmethod
    def _count_tensors(m, only_trainable=True):
        """Count the number of [trainable] tensor objects in a pytorch model.
        """
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return sum([1 for i, l in m.summary.iterrows() for w in l['weight_shapes'] if l['weight_shapes'] and l[k] > 0])

    def count_layers(self, only_trainable=True):
        """Count the number of [trainable] layers in a pytorch model.
        A layer is defined as a module with a nonzero number of [trainable] parameters.
        """
        return self._count_layers(self, only_trainable=only_trainable)
    
    @staticmethod
    def _count_layers(m, only_trainable=True):
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return m.summary[m.summary[k] > 0].shape[0]

    def _initialize_weights(self):
        # Loop in reverse to pick up the nonlinearity following the layer for gain computation
        modules = list(self.modules())
        for m in reversed(modules):
            try:
                gain = calculate_xavier_gain(m.__class__)
            except:
                gain = 1
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                assert gain == calculate_xavier_gain(nn.Conv1d)
                nn.init.xavier_normal(m.weight.data, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                assert gain == calculate_xavier_gain(nn.Linear)
                nn.init.xavier_normal(m.weight.data, gain=gain)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.affine:
                    # Affine transform does nothing at first
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                # if m.track_running_stats:
                # Running stats are initialized to have no history
                m.running_mean.zero_()
                m.running_var.fill_(1)


def transform_range(in_value, in_maxs, in_mins, out_maxs, out_mins):
    """Transform a number from a range into another range, maintaining ratios.
    """
    assert (in_value <= in_maxs).all() and (in_value >= in_mins).all()
    in_range = (in_maxs - in_mins)  
    out_range = (out_maxs - out_mins)  
    return (((in_value - in_mins) * out_range) / in_range) + out_mins


def apply_sigmoid_and_transform(x, **kwargs):
    """Applies the element wise sigmoid function and transforms output into a given range
    """
    view_dim = kwargs.pop('view_dim')
    a = nn.Sigmoid()
    y = a(x)
    # y.data = torch.from_numpy(transform_range(y.data.numpy(), **kwargs))
    y = y.view(view_dim)
    return y


class ClassicalControlFNN(AbstractESModel):
    """
    FNN for classical control problems
    """
    def __init__(self, observation_space, action_space):
        super(ClassicalControlFNN, self).__init__()
        if type(action_space) is gym.spaces.Box:
            # Continuous action space:
            # Physical output to be used directly.
            self.out_dim = action_space.shape
            self.n_out = int(np.prod(action_space.shape))
            out_mins = action_space.low if not np.isinf(action_space.low).any() else - np.ones(action_space.shape)
            out_maxs = action_space.high if not np.isinf(action_space.high).any() else np.ones(action_space.shape)
            sigmoid_mins = - np.ones(out_mins.shape)
            sigmoid_maxs = np.ones(out_maxs.shape)
            trsf_in = {'view_dim': (-1, *self.out_dim), 'in_maxs': sigmoid_maxs, 'in_mins': sigmoid_mins, 'out_maxs': out_maxs, 'out_mins': out_mins}
            self.out_activation = partial(apply_sigmoid_and_transform, **trsf_in)
        elif type(action_space) is gym.spaces.Discrete:
            # Discrete action space: 
            # Probabilistic output to be indexed by maximum probability.
            # Output is the index of the most likely action.
            self.n_out = action_space.n
            self.out_activation = nn.LogSoftmax(dim=1)
        elif type(action_space) is gym.spaces.MultiDiscrete:
            IPython.embed()
            pass
        elif type(action_space) is gym.spaces.MultiBinary:
            IPython.embed()
            pass
        elif type(action_space) is gym.spaces.tuple:
            # Tuple of different action spaces
            # https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py
            IPython.embed()
            pass

        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 1
        assert hasattr(action_space, 'shape')
        self.in_dim = observation_space.shape
        self.n_in = int(np.prod(observation_space.shape))
        self.lin1 = nn.Linear(self.n_in, 32)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.lin6 = nn.Linear(64, 32)
        self.relu6 = nn.ReLU()
        self.lin7 = nn.Linear(32, self.n_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        x = self.relu3(self.lin3(x))
        x = self.relu4(self.lin4(x))
        x = self.relu5(self.lin5(x))
        x = self.relu6(self.lin6(x))
        x = self.out_activation(self.lin7(x))
        return x


class ClassicalControlRNN(AbstractESModel):
    """
    RNN for classical control problems
    """
    def __init__(self, observation_space, action_space):
        super(ClassicalControlRNN, self).__init__()
        if type(action_space) is gym.spaces.Box:
            # Continuous action space:
            # Physical output to be used directly.
            self.out_dim = action_space.shape
            self.n_out = int(np.prod(action_space.shape))
            out_mins = action_space.low if not np.isinf(action_space.low).any() else - np.ones(action_space.shape)
            out_maxs = action_space.high if not np.isinf(action_space.high).any() else np.ones(action_space.shape)
            sigmoid_mins = - np.ones(out_mins.shape)
            sigmoid_maxs = np.ones(out_maxs.shape)
            trsf_in = {'view_dim': (-1, *self.out_dim), 'in_maxs': sigmoid_maxs, 'in_mins': sigmoid_mins, 'out_maxs': out_maxs, 'out_mins': out_mins}
            self.out_activation = partial(apply_sigmoid_and_transform, **trsf_in)
        elif type(action_space) is gym.spaces.Discrete:
            # Discrete action space: 
            # Probabilistic output to be indexed by maximum probability.
            # Output is the index of the most likely action.
            self.n_out = action_space.n
            self.out_activation = nn.LogSoftmax(dim=1)

        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 1
        assert hasattr(action_space, 'shape')
        
        self.n_in = int(np.prod(observation_space.shape))
        self.lin1 = nn.Linear(self.n_in, 32)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(32, self.n_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        x = self.relu3(self.lin3(x))
        x = self.relu4(self.lin4(x))
        x = self.out_activation(self.lin5(x))
        return x


class MujocoFNN(AbstractESModel):
    """
    FNN for Mujoco control problems
    """
    def __init__(self, observation_space, action_space):
        super(MujocoFNN, self).__init__()
        if type(action_space) is gym.spaces.Box:
            # Continuous action space:
            # Physical output to be used directly.
            self.out_dim = action_space.shape
            self.n_out = int(np.prod(action_space.shape))
            out_mins = action_space.low if not np.isinf(action_space.low).any() else - np.ones(action_space.shape)
            out_maxs = action_space.high if not np.isinf(action_space.high).any() else np.ones(action_space.shape)
            sigmoid_mins = - np.ones(out_mins.shape)
            sigmoid_maxs = np.ones(out_maxs.shape)
            trsf_in = {'view_dim': (-1, *self.out_dim), 'in_maxs': sigmoid_maxs, 'in_mins': sigmoid_mins, 'out_maxs': out_maxs, 'out_mins': out_mins}
            self.out_activation = partial(apply_sigmoid_and_transform, **trsf_in)
        elif type(action_space) is gym.spaces.Discrete:
            # Discrete action space: 
            # Probabilistic output to be indexed by maximum probability.
            # Output is the index of the most likely action.
            self.n_out = action_space.n
            self.out_activation = nn.LogSoftmax(dim=1)
        elif type(action_space) is gym.spaces.MultiDiscrete:
            IPython.embed()
            pass
        elif type(action_space) is gym.spaces.MultiBinary:
            IPython.embed()
            pass
        elif type(action_space) is gym.spaces.tuple:
            # Tuple of different action spaces
            # https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py
            IPython.embed()
            pass

        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 1
        assert hasattr(action_space, 'shape')
        self.in_dim = observation_space.shape
        self.n_in = int(np.prod(observation_space.shape))
        self.lin1 = nn.Linear(self.n_in, 512)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(512, 1024)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.lin6 = nn.Linear(256, 128)
        self.relu6 = nn.ReLU()
        self.lin7 = nn.Linear(128, self.n_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        x = self.relu3(self.lin3(x))
        x = self.relu4(self.lin4(x))
        x = self.relu5(self.lin5(x))
        x = self.relu6(self.lin6(x))
        x = self.out_activation(self.lin7(x))
        return x


class DQN(AbstractESModel):
    """The CNN used by Mnih et al (2015) in "Human-level control through deep reinforcement learning" for Atari environments
    """
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(observation_space.shape) == 3
        assert hasattr(action_space, 'n')
        self.in_dim = observation_space.shape
        in_channels = observation_space.shape[0]
        out_dim = action_space.n
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv2_relu = nn.ReLU()
        self.conv3 = nn.Conv2d(64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.conv3_relu = nn.ReLU()
        n_size = self._get_conv_output(observation_space.shape)
        self.lin1 = nn.Linear(n_size, 512)
        self.lin1_relu = nn.ReLU()
        self.lin2 = nn.Linear(512, out_dim)
        self.lin2_logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.lin1_relu(self.lin1(x))
        x = self.lin2_logsoftmax(self.lin2(x))
        return x

    def _get_conv_output(self, shape):
        """Compute the number of output parameters from convolutional part by forward pass
        """
        bs = 1
        inputs = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(inputs)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv1_relu(self.conv1(x))
        x = self.conv2_relu(self.conv2(x))
        x = self.conv3_relu(self.conv3(x))
        return x


class MNISTNet(AbstractESModel):
    """ 
    Convolutional neural network for use on the MNIST data set.

    It uses batch normalization to normalize layer outputs before 
    applying pooling and nonlinearity according to Ioffe (2015) [https://arxiv.org/pdf/1502.03167.pdf]
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.in_dim = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv1_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2_relu = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(50, 10)
        self.fc2_logsoftmax = nn.LogSoftmax(dim=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1_relu(self.conv1_pool(self.conv1_bn(self.conv1(x))))
        x = self.conv2_relu(self.conv2_pool(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_logsoftmax(self.fc2(x))
        return x


class MNISTNetDropout(AbstractESModel):
    """ 
    Convolutional neural network for use on the MNIST data set.

    It uses batch normalization to normalize layer outputs before 
    applying pooling and nonlinearity according to Ioffe (2015) [https://arxiv.org/pdf/1502.03167.pdf]
    """
    def __init__(self):
        super(MNISTNetDropout, self).__init__()
        self.in_dim = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv1_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2_relu = nn.ReLU()
        self.conv2_dropout = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(320, 50)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(50, 10)
        self.fc2_logsoftmax = nn.LogSoftmax(dim=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1_relu(self.conv1_pool(self.conv1(x)))
        x = self.conv2_dropout(self.conv2_relu(self.conv2_pool(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.fc1_dropout(self.fc1_relu(self.fc1(x)))
        x = self.fc2_logsoftmax(self.fc2(x))
        return x


class MNISTNetNoBN(AbstractESModel):
    """This version uses no batch normalization
    """
    def __init__(self):
        super(MNISTNetNoBN, self).__init__()
        self.in_dim = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv1_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2_relu = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(50, 10)
        self.fc2_logsoftmax = nn.LogSoftmax(dim=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1_relu(self.conv1_pool(self.conv1(x)))
        x = self.conv2_relu(self.conv2_pool(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_logsoftmax(self.fc2(x))
        return x


class MNISTNetNoInit(MNISTNet):
    """This version uses default weight initialization
    """
    def __init__(self):
        super(MNISTNetNoInit, self).__init__()
        self.in_dim = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv1_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2_relu = nn.ReLU()

        self.fc1 = nn.Linear(320, 50)
        self.fc1_relu = nn.ReLU()

        self.fc2 = nn.Linear(50, 10)
        self.fc2_logsoftmax = nn.LogSoftmax(dim=1)


class CIFARNet(nn.Module):
    """A CNN for the CIFARn data set where n is variable.
    http://www.isip.uni-luebeck.de/fileadmin/uploads/tx_wapublications/hertel_ijcnn_2015.pdf

    No. Layer           Dimension               Kernel  Stride  Padding
                        Width   Height  Depth
    0   Input           227     227     3       -       -       -
    1   Convolution     55      55      96      11      4       -
    2   Relu            55      55      96      -       -       -
    3   Pooling         27      27      96      3       2       -
    4   Normalization   27      27      96      -       -       -

    5   Convolution     27      27      256     5       1       2
    6   Relu            27      27      256     -       -       -
    7   Pooling         13      13      256     3       2       -
    8   Normalization   13      13      256     -       -       -

    9   Convolution     13      13      384     3       1       1
    10  Relu            13      13      384     -       -       -

    11  Convolution     13      13      384     3       1       1
    12  Relu            13      13      384     -       -       -

    13  Convolution     13      13      256     3       1       1
    14  Relu            13      13      256     -       -       -
    15  Pooling         6       6       256     3       2       -

    16  Fully Connected 1       1       4096    -       -       -
    17  Relu            1       1       4096    -       -       -
    18  Dropout         1       1       4096    -       -       -

    19  Fully Connected 1       1       4096    -       -       -
    20  Relu            1       1       4096    -       -       -
    21  Dropout         1       1       4096    -       -       -

    22  Fully Connected 1       1       1000    -       -       -
    23  Softmax         1       1       1000    -       -       -
    """
    
    def __init__(self, n=10):
        super(CIFARNet, self).__init__()

        IPython.embed()
        self.in_dim = torch.Size((1, 28, 28))
        self.conv1 = nn.Conv2d(3, 96, (11, 11), stride=4)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv1_pool = nn.MaxPool2d((3,3), stride=2)
        self.conv1_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(27, 256, (5, 5), stride=4)
        self.conv1_bn = nn.BatchNorm2d(256)
        self.conv2_pool = nn.MaxPool2d((3,3), stride=2)
        self.conv2_relu = nn.ReLU()

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self._initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = capsule_softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, n_classes):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=n_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * n_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(n_classes)).cuda().index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
