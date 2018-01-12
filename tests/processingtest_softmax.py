import time
import IPython
import gym
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

    def __init__(self, observation_space, action_space, acti='relu'):
        super(FFN, self).__init__()
        assert hasattr(observation_space, 'shape') and len(
            observation_space.shape) == 1
        assert hasattr(action_space, 'n')
        in_dim = observation_space.shape[0]
        out_dim = action_space.n
        self.lin1 = nn.Linear(in_dim, 32)
        self.lin2 = nn.Linear(32, 64)
        self.lin3 = nn.Linear(64, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, out_dim)
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


def gym_rollout(max_episode_length, model, random_seed, env, is_antithetic):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(),
                     requires_grad=True).unsqueeze(0)
    retrn = 0
    nsteps = 0
    done = False
    # Rollout
    while not done and nsteps < max_episode_length:
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        nsteps += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(),
                         requires_grad=True).unsqueeze(0)


def gym_rollout_multi(max_episode_length, model, random_seed, return_queue, env, is_antithetic):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(),
                     requires_grad=True).unsqueeze(0)
    retrn = 0
    nsteps = 0
    done = False
    # Rollout
    while not done and nsteps < max_episode_length:
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data.numpy()
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        nsteps += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(),
                         requires_grad=True).unsqueeze(0)
    return_queue.put({'seed': random_seed, 'return': retrn,
                      'is_anti': is_antithetic, 'nsteps': nsteps})


# SINGLE THREADED
for acti in ['relu', 'softmax']:
    n = 25000
    env = gym.make('CartPole-v0')
    model = FFN(env.observation_space, env.action_space, acti=acti)
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(),requires_grad=True).unsqueeze(0)   

    ts = time.perf_counter()
    for i in range(n):
        model(state)
        #gym_rollout(1000, model, 'dummy_seed', env, 'dummy_neg')
    tf = time.perf_counter()

    print("single " + acti + " total: " + str(tf-ts))
    print("single " + acti + " per call: " + str((tf - ts) / n) + "\n")


# MULTI THREADED
n_parallel = 10
n_forwards = 2500
def do_forward(model, x):
    for i in range(n_forwards):
        model(x)

for acti in ['relu', 'softmax']:

    env = gym.make('CartPole-v0')
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)  
    return_queue = mp.Queue()

    models = []
    #print("0")
    for i in range(n_parallel):
        models.append(FFN(env.observation_space, env.action_space, acti=acti))
    #print("1")

    #IPython.embed()
    # models[0](state)

    processes = []
    ts = time.perf_counter()
    for i in range(n_parallel):
        p = mp.Process(target=do_forward, args=(models[i], state))
        #p = mp.Process(target=gym_rollout_multi, args=(1000, models[i], 'dummy_seed', return_queue, env, 'dummy_neg'))
        p.start()
        processes.append(p)
    #print("2")
    # Force join
    for p in processes:
        p.join()
    #print("3")
    # Get results of finished processes
    #raw_output = [return_queue.get() for p in processes]
    tf = time.perf_counter()
    print("multi " + acti + " total: " + str(tf-ts))
    print("multi " + acti + " per call: " + str((tf-ts)/(n_forwards*n_parallel)) + "\n")
    #print("4")
    #print(raw_output)
