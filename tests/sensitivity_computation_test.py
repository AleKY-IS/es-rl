import platform
import time

import IPython
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from context import es
from es.train import compute_gradients
from es.train import compute_sensitivities


class SingleLayerNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleLayerNet, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        return x


class TwoLayerNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleLayerNet, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        return x


def test_single_layer(testid, batch_size, in_dim, out_dim, do_debug=False):
    """
    Test for 1D input and 1D output with 1 weight and 1 bias.
    y = p0 * x + p1
    """
    if do_debug:
        IPython.embed()
    # Set seed
    torch.manual_seed(1)
    # Random input
    x = Variable(torch.randn(batch_size, in_dim), requires_grad=True)
    # Init model
    model = SingleLayerNet(in_dim=in_dim, out_dim=out_dim)
    # Model output
    model_y = model(x)
    # Sensitivities dy/dw
    compute_sensitivities(model, x)

    # Controls (analytical, see notes)
    control_y = (model.lin1.weight @ x.transpose(0, -1)).transpose(0, -1).add(model.lin1.bias)
    weight_gradient = x.sum(0).repeat(out_dim, 1)  # Can repeat for single layer network since outputs are linearly independent
    bias_gradient = Variable(torch.ones(out_dim)*batch_size)
    control_p_grad = [weight_gradient, bias_gradient]
    
    # Print values
    print("TEST " + str(testid))
    print(" - batch_size = " + str(batch_size))
    print(" - in_dim = " + str(in_dim))
    print(" - out_dim = " + str(out_dim))
    print("x = " + str(x.data.numpy()))
    for idx, param in enumerate(model.parameters()):
        print("p" + str(idx) + " = " + str(param.data.numpy()))
    print("model y = " + str(model_y.data.numpy()))
    print("control y = " + str(control_y.data.numpy()))
    for idx, param in enumerate(model.parameters()):
        method_p_grad = param.grad.data.numpy()
        print("method p" + str(idx) + "_sens = " + str(method_p_grad))
        print("control p" + str(idx) + "_sens = " + str(control_p_grad[idx].data.numpy()))

    # Print checks
    is_test_passed = True
    if (np.abs(model_y.data.numpy() - control_y.data.numpy()) < 1e-4).all():
        print("✓ Model and control output match!")
    else:
        print("✗ Model and control output do not match!")
        is_test_passed = False
    for idx, param in enumerate(model.parameters()):
        method_p_grad = param.grad.data.numpy()
        if (np.abs(method_p_grad - control_p_grad[idx].data.numpy()) < 1e-4).all():
            print("✓ Model and control sensitivities match!")
        else:
            print("✗ Model and control sensitivities do not match!")
            is_test_passed = False
    impliesarrow = '\u21D2'
    if is_test_passed:
        print(impliesarrow + " TEST " + str(testid) + " passed!")
    else:
        print(impliesarrow + " TEST " + str(testid) + " not passed!")
    print("============================================================")


if __name__ == '__main__':
    # 1D INPUT AND OUTPUT, y = ax+b
    test_single_layer(testid=1, batch_size=1, in_dim=1, out_dim=1)
    test_single_layer(testid=2, batch_size=1, in_dim=2, out_dim=1)
    test_single_layer(testid=3, batch_size=1, in_dim=1, out_dim=2)
    test_single_layer(testid=4, batch_size=1, in_dim=2, out_dim=3)
    test_single_layer(testid=5, batch_size=2, in_dim=2, out_dim=3)
    test_single_layer(testid=6, batch_size=10, in_dim=2, out_dim=3)
    test_single_layer(testid=7, batch_size=2, in_dim=10, out_dim=10)



