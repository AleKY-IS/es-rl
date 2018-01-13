import platform
import time

import IPython
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class FFN(nn.Module):
    """
    FFN for classical control problems
    """

    def __init__(self, acti='relu'):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(10, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 10)
        self.acti = acti

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        
        if self.acti == 'softmax':
            x = F.log_softmax(self.lin5(x), dim=0)
        elif self.acti == 'relu':
            x = F.relu(self.lin5(x))
        return x


if __name__ == '__main__':
    # Model input
    n = 25000
    state = Variable(torch.randn(10), requires_grad=True)
    

    # SINGLE THREADED
    for acti in ['relu', 'softmax']:
        model = FFN(acti=acti)
        ts = time.perf_counter()
        for i in range(n):
            model(state)
        tf = time.perf_counter()

        print("single " + acti + " total: " + str(tf-ts))
        print("single " + acti + " per call: " + str((tf - ts) / n) + "\n")


    # MULTI THREADED
    # Set number of threads for CPU parallelized operations
    if platform.system() == 'Linux':
        torch.set_num_threads(1)
    print("Num threads = " + str(torch.get_num_threads()))

    # Method to do multiple forward passes (analogous to rollout)
    def do_forward(model, x):
        for i in range(n_forwards):
            model(x)

    n_parallel = 10
    n_forwards = int(n/n_parallel)
    for acti in ['relu', 'softmax']:
        # Initialize return queue
        return_queue = mp.Queue()
        # Intialize models
        models = []
        for i in range(n_parallel):
            models.append(FFN(acti=acti))

        # Create and start processes
        processes = []
        ts = time.perf_counter()
        for i in range(n_parallel):
            p = mp.Process(target=do_forward, args=(models[i], state))

            #time.sleep(1)
            p.start()
            processes.append(p)

        # Force join
        for p in processes:
            p.join()
        
        tf = time.perf_counter()
        print("multi " + acti + " total: " + str(tf-ts))
        print("multi " + acti + " per call: " + str((tf-ts)/(n_forwards*n_parallel)) + "\n")
