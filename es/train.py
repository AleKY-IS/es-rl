import math
import os
import queue
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim
import torch.legacy.optim as legacyOptim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable

from torch import nn
import gym


def print_init(args):
    if print_every > 0:
        print("----- Evolver parameters ------")
        print("Environment:          %s" % args.env_name)
        print("Population:           %s" % args.model_name)
        print("Workers:              %d" % args.agents)
        print("Generations:          %d" % args.max_generations)
        print("Sigma:                %5.4f" % args.sigma)
        print("Learning rate:        %5.4f\n" % args.lr)
        print("Learning rate decay:  %5.4f\n" % args.lr-decay)
        print("Restore point:        %s\n" % args.restore)
        print("------- Running evolver -------")


def print_iter(args, gen, prtint=1):
    if prtint and gen % prtint == 0:
        print("Generation: {:3d} | Reward {: 4.1f} | Time {:4.2f} seconds".format(gen, test_reward, t))


def unperturbed_rank(returns, unperturbed_results):
    nth_place = 1
    for r in returns:
        if r > unperturbed_results:
            nth_place += 1
    rank_diag = '{:d} out of {:d}'.format(nth_place, len(returns) + 1)
    return rank_diag, nth_place


def perturb_model(args, parent_model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the mirrored perturbation, and returns both perturbed
    models.
    """
    # Get model class and instantiate two new models as copies of parent
    model_class = type(parent_model)
    model1 = model_class(env.observation_space, env.action_space) if hasattr(env, 'observation_space') else model_class()
    model2 = model_class(env.observation_space, env.action_space) if hasattr(env, 'observation_space') else model_class()
    model1.load_state_dict(parent_model.state_dict())
    model2.load_state_dict(parent_model.state_dict())
    np.random.seed(random_seed)
    #IPython.embed()
    # Permute all weights of each model by isotropic Gaussian noise
    for param1, param2, pp in zip(model1.es_parameters(), model2.es_parameters(), parent_model.es_parameters()):
        eps = torch.from_numpy(np.random.normal(0, 1, pp.data.size())).float()
        eps = eps#/pp.grad.data  # Scale by sensitivities
        eps = eps#/eps.std()     # Rescale to unit variance
        param1.data += args.sigma*eps
        param2.data -= args.sigma*eps

    return [model1, model2]


def generate_seeds_and_models(args, parent_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(args, parent_model, random_seed, env)
    return random_seed, two_models


def fitness_shaping(returns):
    """
    Performs the fitness rank transformation used for CMA-ES.
    Reference: Natural Evolution Strategies [2014]
    """
    n = len(returns)
    sorted_indices = np.argsort(-np.array(returns))
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
    return u/np.sum(u)-1/n


def compute_sensitivities(parent_model, inputs):
    #IPython.embed()
    parent_model.zero_grad()
    outputs = parent_model(inputs)
    batch_size = outputs.data.size()[0]
    n_outputs = outputs.data.size()[1]
    do_square = False
    do_square_root = True
    do_abs = False
    do_squash = False
    do_numerical = False
    # Loop over output units (and sum by accumulation of gradients)
    sensitivities = []
    for idx in range(n_outputs):
        parent_model.zero_grad()
        t = torch.zeros(batch_size, n_outputs)
        t[:, idx] = torch.ones(batch_size, 1)
        # Compute dy_t/dw on batch
        outputs.backward(t, retain_graph=True)
        # Get computed sensitivities
        for pid, param in enumerate(parent_model.parameters()):
            if idx == 0:
                sensitivities.append(param.grad.data.pow(2))
            else:
                sensitivities[pid] += param.grad.data.pow(2)

        if do_abs: # 
            for param in parent_model.parameters():
                param.grad = param.grad.abs()

    if do_square: # (SM-G-SUM)
        for param in parent_model.parameters():
            param.grad = param.grad.pow(2)

    if do_square_root: # (SM-G-SUM)
        for pid in range(len(sensitivities)):
            sensitivities[pid] = sensitivities[pid].sqrt()
        # for param in parent_model.parameters():
        #     param.grad = param.grad.sqrt()

    if do_squash:
        for param in parent_model.parameters():
            param.grad = param.grad/param.grad.max()

    # Set gradients
    for pid, param in enumerate(parent_model.parameters()):
        param.grad.data = sensitivities[pid]
    
    # Numerical considerations
    if do_numerical:
        for param in parent_model.parameters():
            # Set any sensitivities of zero to one to avoid dividing by zero
            #param.grad.data[param.grad.data == 0] = 1
            # Add small number so as not to divide by extremely small numbers
            param.grad.data += 10**-3
            param.grad.data[param.grad.data > 10**3] = 10**3

    # mx = 0
    # mn = 10**10
    # for param in parent_model.parameters():
    #     mx = param.grad.data.max() if param.grad.data.max() > mx else mx
    #     mn = param.grad.data.max() if param.grad.data.max() < mn else mn
    # print(mx)
    # print(mn)
    return parent_model


def compute_gradients(args, parent_model, returns, random_seeds, is_anti_list):
    # Verify input
    batch_size = len(returns)
    assert batch_size == args.agents
    assert len(random_seeds) == batch_size

    # Shape returns and get rank of unperturbed model
    shaped_returns = fitness_shaping(returns)


    # Preallocate list with gradients
    gradients = []
    for param in parent_model.parameters():
        gradients.append(torch.zeros(param.data.size()))

    # Compute gradients
    # TODO: Always use an optimizer to step on the computed gradients [DONE]
    for i in range(args.agents):
        # Set random seed, get antithetic multiplier and reward
        np.random.seed(random_seeds[i])
        multiplier = -1 if is_anti_list[i] else 1
        reward = shaped_returns[i]
        for layer, param in enumerate(parent_model.parameters()):
            eps = np.random.normal(0, 1, param.data.size())
            gradients[layer] += (torch.from_numpy((args.agents*args.sigma) * (reward*multiplier*eps)).float())
            #gradients[layer] += torch.from_numpy(eps).float()

    # Set gradients
    parent_model.zero_grad()
    for layer, param in enumerate(parent_model.parameters()):
        param.grad.data = - gradients[layer]
    

    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    # TODO: Move shared code out of branch
    # TODO: Always use an optimizer:
    #       1. Compute gradients (by some method)
    #       2. Set gradients 
    #               for param in model.parameters():
    #                   param.grad = gradients
    #       3. Optimize with optimizer
    #               optimizer.step()

    # gradient = []
    # for i in range(args.agents):
    #     np.random.seed(random_seeds[i])
    #     multiplier = -1 if is_anti_list[i] else 1
    #     reward = shaped_returns[i]
    #     # Compute gradients
    #     for k, v in parent_model.es_params():
    #         # pytorch in general
    #         loss.backward()
    #         # ES strategy
    #         eps = np.random.normal(0, 1, v.size())
    #         gradient.append(torch.from_numpy((args.agents*args.sigma) * (reward*multiplier*eps)).float())
    #         # Safe mutations
    #         eps = np.random.normal(0, 1, v.size())
    #         (mini-batch of network outputs).backward() # Compute sensitivities and store them in v.grad.data
    #         gradient.append(eps/v.grad.data) # Scale permutations with sensitivities
        
        # Set gradients
        # i = 0
        # for param in parent_model.parameters():
        #     param.grad = gradient[i]
        
        # Optimize parameters
        # optimizer.step()


    # if args.optimizer == 'Adam':
    #     globalGrads = None
    #     for i in range(args.agents):
    #         np.random.seed(random_seeds[i])
    #         multiplier = -1 if is_anti_list[i] else 1
    #         sr = shaped_returns[i]

    #         localGrads = []
    #         idx = 0
    #         for param in parent_model.es_parameters():
    #             eps = np.random.normal(0, 1, param.data.size())
    #             grad = torch.from_numpy((args.agents*args.sigma) * (sr*multiplier*eps)).float()

    #             localGrads.append(grad)
                
    #             if len(optimConfig) == idx:
    #                 optimConfig.append({'learningRate': args.lr})
    #             idx = idx + 1
            
    #         if globalGrads is None:
    #             globalGrads = localGrads
    #         else:
    #             for i in range(len(globalGrads)):
    #                 globalGrads[i] = torch.add(globalGrads[i], localGrads[i])

    #     idx = 0
    #     for param in parent_model.es_parameters():
    #         r, _ = legacyOptim.adam(lambda x:  (1, -globalGrads[idx]), param.data, optimConfig[idx])
    #         param.data.copy_(r)
    #         idx = idx + 1
    # else:
    #     # For each model, generate the same random numbers as we did
    #     # before, and update parameters. We apply weight decay once.
    #     for i in range(args.agents):
    #         np.random.seed(random_seeds[i])
    #         multiplier = -1 if is_anti_list[i] else 1
    #         sr = shaped_returns[i]
    #         for param in parent_model.es_parameters():
    #             eps = np.random.normal(0, 1, param.data.size())
    #             grad = torch.from_numpy((args.agents*args.sigma) * (sr*multiplier*eps)).float()
    #             param.data += args.lr * grad
    #     args.lr *= args.lr_decay

    # IPython.embed()
    return parent_model


# TODO: Examine possibility of reusing pool of workers 
#       - p = Pool(args.agents)
#       - for _ in range(args.max_generations):
#       -   for j in range(int(args.agents/2)):
#       -       inputs.append((args, perturbed_model, seed, return_queue, env, is_negative))
#       -   p.imap_unordered(eval_fun, args=inputs) 
def train_loop(args, parent_model, env, eval_fun, optimizer, chkpt_dir):
    # Initialize list of processes, seeds and models and return queue
    #processes, seeds, models = [], [], []
    # Initialize structures for saving statistics
    reward_unperturbed = []
    reward_average = []
    reward_max = []
    reward_min = []
    reward_var = []
    generations = []
    episodes = []
    observations = []
    walltimes = []

    # 
    return_queue = mp.Queue()
    
    # Evaluate parent model
    eval_fun(args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg', collect_inputs=True)
    unperturbed_out = return_queue.get()
    # Start training loop
    n_episodes = 0
    n_generation = 0
    n_observations = 0
    start_time = time.perf_counter()
    for n_generation in range(args.max_generations):
        loop_start_time = time.perf_counter()
        # Empty list of processes, seeds and models and return queue
        processes, seeds, models = [], [], []
        return_queue = mp.Queue()
        # done = mp.Event()

        # Compute parent model weight-output sensitivities
        compute_sensitivities(parent_model, Variable(torch.from_numpy(unperturbed_out['inputs'])))

        # Generate a perturbation and its antithesis
        # TODO: This could be be part of the parallel execution (somehow)
        for j in range(int(args.agents/2)):
            random_seed, two_models = generate_seeds_and_models(args, parent_model, env)
            # Add twice because we get two models with the same seed
            seeds.append(random_seed)
            seeds.append(random_seed)
            models.extend(two_models)
        assert len(seeds) == len(models)

        # Keep track of which perturbations were positive and negative.
        # Start with negative true because pop() makes us go backwards.
        is_negative = True
        # Add all peturbed models to the queue
        workers_start_time = time.perf_counter()
        while models:
            perturbed_model = models.pop()
            seed = seeds.pop()
            inputs = (args, perturbed_model, seed, return_queue, env, is_negative)
            p = mp.Process(target=eval_fun, args=inputs)
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(seeds) == 0

        # Evaluate the unperturbed model as well
        p = mp.Process(target=eval_fun, args=(args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg'), kwargs={'collect_inputs': True})
        p.start()
        processes.append(p)

        # Get output from processes until all are terminated
        raw_output = []
        while processes:
            # Update live processes
            processes = [p for p in processes if p.is_alive()]
            # Get all returns from finished processes return queue
            while not return_queue.empty():
                raw_output.append(return_queue.get(False))

        # Force join
        for p in processes:
            p.join()

        workers_end_time = time.perf_counter()        
        
        # Split into parts
        seeds = [out['seed'] for out in raw_output]
        returns = [out['return'] for out in raw_output]
        is_anti_list = [out['is_anti'] for out in raw_output]
        i_observations = [out['n_observations'] for out in raw_output]

        # Get results of unperturbed model and remove it from all results
        unperturbed_index = seeds.index('dummy_seed')
        unperturbed_out = raw_output.pop(unperturbed_index)
        assert unperturbed_out['seed'] == 'dummy_seed'
        seeds.pop(unperturbed_index)
        returns.pop(unperturbed_index)
        is_anti_list.pop(unperturbed_index)
        i_observations.pop(unperturbed_index)
        
        # Update gradient
        compute_gradients(args, parent_model, returns, seeds, is_anti_list)
        optimizer.step()
        # lr_scheduler.step()

        if hasattr(args, 'variable_ep_len') and args.variable_ep_len:
            args.max_episode_length = int(5*max(num_frames))

        # Compute rank of unperturbed model
        rank_diag, rank = unperturbed_rank(returns, unperturbed_out['return'])
        
        # Time
        loop_end_time = time.perf_counter()

        # Store statistics
        n_episodes += len(returns)
        n_observations += sum(i_observations)
        generations.append(n_generation)
        episodes.append(n_episodes)
        observations.append(n_observations)
        walltimes.append(loop_end_time - start_time)
        reward_unperturbed.append(unperturbed_out['return'])
        reward_average.append(np.mean(returns))
        reward_max.append(max(returns))
        reward_min.append(min(returns))
        reward_var.append(np.var(returns))
        

        # Plot
        if n_episodes % (40*20) == 0:
            # NOTE: Possible x-axis are: generations, episodes, observations, time
            fig = plt.figure()
            pltUnp, = plt.plot(episodes, reward_unperturbed, label='parent')
            pltAvg, = plt.plot(episodes, reward_average, label='average')
            pltMax, = plt.plot(episodes, reward_max, label='max')
            pltMin, = plt.plot(episodes, reward_min, label='min')
            plt.ylabel('rewards')
            plt.xlabel('episodes')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/progress.pdf')
            plt.close(fig)

            fig = plt.figure()
            walltimes_plot = np.array(walltimes)/60
            pltUnp, = plt.plot(walltimes_plot, reward_unperturbed, label='parent')
            pltAvg, = plt.plot(walltimes_plot, reward_average, label='average')
            pltMax, = plt.plot(walltimes_plot, reward_max, label='max')
            pltMin, = plt.plot(walltimes_plot, reward_min, label='min')
            plt.ylabel('rewards')
            plt.xlabel('time [min]')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/progress2.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltUnp, = plt.plot(observations, reward_unperturbed, label='parent')
            pltAvg, = plt.plot(observations, reward_average, label='average')
            pltMax, = plt.plot(observations, reward_max, label='max')
            pltMin, = plt.plot(observations, reward_min, label='min')
            plt.ylabel('rewards')
            plt.xlabel('observations')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/progress3.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltUnp, = plt.plot(generations, reward_unperturbed, label='parent')
            pltAvg, = plt.plot(generations, reward_average, label='average')
            pltMax, = plt.plot(generations, reward_max, label='max')
            pltMin, = plt.plot(generations, reward_min, label='min')
            plt.ylabel('rewards')
            plt.xlabel('generations')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/progress4.pdf')
            plt.close(fig)

            # TODO: Plot of time used
            # TODO: Plot of reward variance
            # TODO: Plot of smoothed reward
            # TODO: Plot unperturbed rank

        # Save checkpoint
        if n_episodes % (40*20) == 0:
            torch.save(parent_model.state_dict(), os.path.join(chkpt_dir, 'model_state_dict.pth'))
            torch.save(optimizer.state_dict(), os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))

            # DONE: model.state_dict
            # DONE: optimizer.state_dict
            # TODO: lr_scheduler.state_dict
            # TODO: data used in plots and prints (maybe collect this is single data structure? dictionary?)
            # TODO: settings for algorithm (sigma)
            # TODO: Save into file's directory (not callers)
            # TODO: Keep track of best model as well as newest model
            # TODO: Make checkpoint based on time (every 1 minute e.g.)

        # Print to console
        if not args.silent:
            print('Episode num: %d\n'
                'Average reward: %f\n'
                'Variance in rewards: %f\n'
                'Max reward: %f\n'
                'Min reward: %f\n'
                'Batch size: %d\n'
                'Sigma: %f\n'
                'Learning rate: %f\n'
                'Unperturbed reward: %f\n'
                'Unperturbed rank: %s\n' 
                'Optimizer: %s' %
                (episodes[-1], reward_average[-1], reward_var[-1], reward_max[-1],
                reward_min[-1], args.agents, args.sigma, args.lr,
                unperturbed_out['return'], rank_diag, args.optimizer))
            print("Worker time: {}".format(workers_end_time-workers_start_time))
            print("Loop time: {}".format(loop_end_time-loop_start_time))
            print()




















class FFN(nn.Module):
    """
    FFN for classical control problems
    """

    def __init__(self, observation_space, action_space):
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

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        print("beofre")
        print(x)
        x = F.log_softmax(self.lin5(x), dim=1)
        print("did softmax")
        # x = F.relu(self.lin5(x))
        return x

def gym_rollout(max_episode_length, model, random_seed, return_queue, env, is_antithetic):
    """
    Function to do rollouts of a policy defined by `model` in given environment
    """
    # Reset environment
    state = env.reset()
    state = Variable(torch.from_numpy(state).float(),
                    requires_grad=True).unsqueeze(0)
    retrn = 0
    n_observations = 0
    done = False
    # Rollout
    while not done and n_observations < max_episode_length:
        # Choose action
        actions = model(state)
        action = actions.max(1)[1].data.numpy()
        # Step
        state, reward, done, _ = env.step(action[0])
        retrn += reward
        n_observations += 1
        # Cast state
        state = Variable(torch.from_numpy(state).float(),
                        requires_grad=True).unsqueeze(0)
    return_queue.put({'seed': random_seed, 'return': retrn,
                    'is_anti': is_antithetic, 'n_observations': n_observations})

def train_loopasd(args, parent_model, env, eval_fun, chkpt_dir):
    for _ in range(args.max_generations):
        env = gym.make('CartPole-v0')
        return_queue = mp.Queue()

        models = []
        for i in range(10):
            models.append(FFN(env.observation_space, env.action_space))

        processes = []
        for i in range(10):
            p = mp.Process(target=gym_rollout, args=(
                1000, models[i], 'dummy_seed', return_queue, env, 'dummy_neg'))
            p.start()
            processes.append(p)

        # Force join
        for p in processes:
            p.join()

        # Get results of finished processes
        raw_output = [return_queue.get() for p in processes]
        





