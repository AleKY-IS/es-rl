import math
import os
import pickle
import queue
import time

import gym
import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.legacy.optim as legacyOptim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim
from torch import nn
from torch.autograd import Variable
from .utils import get_lr, plot_stats, save_checkpoint, print_init, print_iter


def unperturbed_rank(returns, unperturbed_return):
    """
    Returns the rank of the unperturbed model among the pertubations.
    """
    nth_place = (returns > unperturbed_return).sum() + 1
    rank_diag = '{:d} out of {:d}'.format(nth_place, len(returns) + 1)
    return rank_diag, nth_place


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
    sorted_indices = np.argsort(-returns)
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n/2+1)-np.log(k+1)])
    return u/np.sum(u)-1/n


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
    model1.load_state_dict(parent_model.state_dict())  # This does not load the gradients (doesn't matter here though)
    model2.load_state_dict(parent_model.state_dict())
    model1.zero_grad()
    model2.zero_grad()
    np.random.seed(random_seed)
    # Permute all weights of each model by isotropic Gaussian noise
    for param1, param2, pp in zip(model1.es_parameters(), model2.es_parameters(), parent_model.es_parameters()):
        eps = torch.from_numpy(np.random.normal(0, 1, pp.data.size())).float()
        eps = eps/pp.grad.data    # Scale by sensitivities
        eps = eps/eps.std()       # Rescale to zero mean unit variance
        param1.data += args.sigma*eps
        param2.data -= args.sigma*eps
        assert not np.isnan(param1.data).any()
        assert not np.isinf(param1.data).any()
        assert not np.isnan(param2.data).any()
        assert not np.isinf(param2.data).any()
    return [model1, model2]


def compute_sensitivities(parent_model, inputs):
    """
    Computes the output-weight sensitivities of the model given a mini-batch
    of inputs.
    Currently implements the SM-G-ABS version of the sensitivities.
    Reference: Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients [2017]
    """
    # Forward pass on input batch
    outputs = parent_model(inputs)
    batch_size = outputs.data.size()[0]
    n_outputs = outputs.data.size()[1]
    do_square = True
    do_square_root = True
    do_abs = False
    do_normalize = True
    do_numerical = True
    # Backward pass for each output unit (and accumulate gradients)
    sensitivities = []
    for idx in range(n_outputs):
        parent_model.zero_grad()
        t = torch.zeros(batch_size, n_outputs)
        t[:, idx] = torch.ones(batch_size, 1)
        # Compute dy_t/dw on batch
        outputs.backward(t, retain_graph=True)
        # Get computed sensitivities and sum into those of other output units
        for pid, param in enumerate(parent_model.parameters()):
            sens = param.grad.data.clone()  # Clone to sum correctly
            if do_square:
                sens = sens.pow(2)
            if do_abs:
                sens = sens.abs()
            if idx == 0:
                sensitivities.append(sens)
            else:
                sensitivities[pid] += sens

    if do_square_root:
        for pid in range(len(sensitivities)):
            sensitivities[pid] = sensitivities[pid].sqrt()
    
    # Normalize
    if do_normalize:
        for pid in range(len(sensitivities)):
            sensitivities[pid] = sensitivities[pid]/sensitivities[pid].max()
    
    # Numerical considerations
    if do_numerical:
        for pid in range(len(sensitivities)):
            # Absolutely insensitive parameters are unscaled
            sensitivities[pid][sensitivities[pid] < 1e-5] = 1
            # Clip sensitivities at a large constant value
            sensitivities[pid][sensitivities[pid] > 1e5] = 1e5

    # Set gradients
    parent_model.zero_grad()
    for pid, param in enumerate(parent_model.parameters()):
        param.grad.data = sensitivities[pid].clone()
        assert not np.isnan(param.grad.data).any()
        assert not np.isinf(param.grad.data).any()
    return parent_model


def compute_gradients(args, parent_model, returns, random_seeds, is_anti_list):
    """
    Computes the gradients of the weights of the model wrt. to the return. 
    The gradients will point in the direction of change in the weights resulting in a 
    decrease in the return.
    """
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
    # - ES strategy
    for i in range(args.agents):
        # Set random seed, get antithetic multiplier and return
        np.random.seed(random_seeds[i])
        multiplier = -1 if is_anti_list[i] else 1
        retrn = shaped_returns[i]
        for layer, param in enumerate(parent_model.parameters()):
            eps = torch.from_numpy(np.random.normal(0, 1, param.data.size())).float()
            eps = eps/param.grad.data   # Scale by sensitivities
            eps = eps/eps.std()         # Rescale to zero mean unit 
            gradients[layer] += 1/(args.agents*args.sigma) * (retrn*multiplier*eps) 
            #sigma_gradient += 

    # Set gradients
    parent_model.zero_grad()
    for layer, param in enumerate(parent_model.parameters()):
        param.grad.data = - gradients[layer]

    return parent_model


# TODO: Examine possibility of reusing pool of workers 
#       - p = Pool(args.agents)
#       - for _ in range(args.max_generations):
#       -   for j in range(int(args.agents/2)):
#       -       inputs.append((args, perturbed_model, seed, return_queue, env, is_negative))
#       -   p.imap_unordered(eval_fun, args=inputs) 
def train_loop(args, parent_model, env, eval_fun, optimizer, lr_scheduler, chkpt_dir, stats=None, checkpoint_interval=10):
    # Initialize iteration variables and statistics
    if stats is None:
        # Initialize dict for saving statistics
        stat_names = ['generations', 'episodes', 'observations', 'walltimes',
                      'return_avg', 'return_var', 'return_max', 'return_min',
                      'return_unp', 'unp_rank', 'sigma', 'lr', 'start_time']
        stats = {}
        for n in stat_names:
            stats[n] = []
            n_episodes = 0
            n_observations = 0
        start_generation = 0
        stats['start_time'] = time.time()
        print_init(args, parent_model, optimizer, lr_scheduler)
    else:
        # Retrieve info on last iteration
        n_episodes = stats['episodes'][-1]
        n_observations = stats['observations'][-1]
        start_generation = stats['generations'][-1] + 1
    
    # Initialize return queue for multiprocessing
    return_queue = mp.Queue()
    
    # Parameterize noise variance to ensure positivity
    beta = 2*np.log(args.sigma)  # sigma**2 = np.exp(beta)

    # Evaluate parent model
    eval_fun(args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg', collect_inputs=True)
    unperturbed_out = return_queue.get()
    max_unperturbed_return = -1e8
    # Start training loop
    last_checkpoint_time = time.time()
    for n_generation in range(start_generation, args.max_generations):
        # Empty list of processes, seeds and models and return queue
        loop_start_time = time.time()
        processes, seeds, models = [], [], []
        return_queue = mp.Queue()

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
        workers_start_time = time.time()
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
        inputs = (args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg')
        p = mp.Process(target=eval_fun, args=inputs, kwargs={'collect_inputs': True})
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
        workers_end_time = time.time()
        
        # Split into parts
        seeds = [out['seed'] for out in raw_output]
        returns = [out['return'] for out in raw_output]
        is_anti_list = [out['is_anti'] for out in raw_output]
        i_observations = [out['n_observations'] for out in raw_output]
        # Get results of unperturbed model
        unperturbed_index = seeds.index('dummy_seed')
        unperturbed_out = raw_output.pop(unperturbed_index)
        assert unperturbed_out['seed'] == 'dummy_seed'
        # Remove unperturbed results from all results
        seeds.pop(unperturbed_index)
        returns.pop(unperturbed_index)
        is_anti_list.pop(unperturbed_index)
        i_observations.pop(unperturbed_index)
        # Cast to numpy
        returns = np.array(returns)
        
        # Compute gradients, update parameters and learning rate
        stats['lr'].append(get_lr(optimizer))
        compute_gradients(args, parent_model, returns, seeds, is_anti_list)
        optimizer.step()
        if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            #lr_scheduler.step(unperturbed_out['return'])
            # TODO Check that this steps correctly (it steps every patience times and what if returns are negative)
            lr_scheduler.step(returns.mean())
        else:
            lr_scheduler.step()

        # Compute rank of unperturbed model
        rank_diag, rank = unperturbed_rank(returns, unperturbed_out['return'])
        
        # Keep track of best model
        if unperturbed_out['return'] >= max_unperturbed_return:
            best_model_stdct = parent_model.state_dict()
            best_optimizer_stdct = optimizer.state_dict()
            # TODO: Also save stats in "best" version
            max_unperturbed_return = unperturbed_out['return']

        # Store statistics
        # TODO Fix restoring and resuming
        n_episodes += len(returns)
        n_observations += sum(i_observations)
        stats['generations'].append(n_generation)
        stats['episodes'].append(n_episodes)
        stats['observations'].append(n_observations)
        stats['walltimes'].append(time.time() - stats['start_time'])
        stats['return_avg'].append(returns.mean())
        stats['return_var'].append(returns.var())
        stats['return_max'].append(returns.max())
        stats['return_min'].append(returns.min())
        stats['return_unp'].append(unperturbed_out['return'])
        stats['unp_rank'].append(rank)
        stats['sigma'].append(np.exp(0.5*beta))
        
        # Adjust max length of episodes
        if hasattr(args, 'not_var_ep_len') and not args.not_var_ep_len:
            args.batch_size = int(5*max(i_observations))

        # Save checkpoint every `checkpoint_interval` seconds
        if last_checkpoint_time < time.time() - checkpoint_interval:
            plot_stats(stats, chkpt_dir)
            save_checkpoint(parent_model, optimizer, best_model_stdct, best_optimizer_stdct, stats, chkpt_dir)
            last_checkpoint_time = time.time()

        # Print to console
        if not args.silent:
            print_iter(args, stats, workers_start_time, workers_end_time, loop_start_time)


        
        