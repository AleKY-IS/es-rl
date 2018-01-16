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
from .utils import get_lr


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
    model1.zero_grad()
    model2.zero_grad()
    np.random.seed(random_seed)
    # Permute all weights of each model by isotropic Gaussian noise
    for param1, param2, pp in zip(model1.es_parameters(), model2.es_parameters(), parent_model.es_parameters()):
        eps = torch.from_numpy(np.random.normal(0, 1, pp.data.size())).float()
        eps = eps#/pp.grad.data  # Scale by sensitivities
        #eps = (eps-eps.mean())/eps.std()     # Rescale to zero mean unit variance
        #print(eps)
        param1.data += args.sigma*eps
        param2.data -= args.sigma*eps
    return [model1, model2]


def compute_sensitivities(parent_model, inputs):
    # Forward pass on input batch
    outputs = parent_model(inputs)
    batch_size = outputs.data.size()[0]
    n_outputs = outputs.data.size()[1]
    do_square = True
    do_square_root = True
    do_abs = False
    do_normalize = False
    # do_squash = False
    do_numerical = False
    # Backward pass for each output unit (and accumulate gradients)
    sensitivities = []
    for idx in range(n_outputs):
        parent_model.zero_grad()
        t = torch.zeros(batch_size, n_outputs)
        t[:, idx] = torch.ones(batch_size, 1)
        # Compute dy_t/dw on batch
        outputs.backward(t, retain_graph=True)
        # Get computed sensitivities
        for pid, param in enumerate(parent_model.parameters()):
            sens = param.grad.data.clone()  # Clone to sum correctly
            if do_square:
                sens = sens.pow(2)
            if do_abs:
                sens = sens.abs()
            # sens = sens.pow(2) if do_square else sens
            # sens = sens.abs() if do_abs else sens
            if idx == 0:
                sensitivities.append(sens)
            else:
                sensitivities[pid] += sens

        # if do_abs:
        #     for param in parent_model.parameters():
        #         param.grad = param.grad.abs()


    if do_square_root: # (SM-G-SUM)
        for pid in range(len(sensitivities)):
            sensitivities[pid] = sensitivities[pid].sqrt()
        # for param in parent_model.parameters():
        #     param.grad = param.grad.sqrt()

    # if do_squash:
    #     for param in parent_model.parameters():
    #         param.grad = param.grad/param.grad.max()

    # if do_square: # (SM-G-SUM)
    #     for param in parent_model.parameters():
    #         param.grad = param.grad.pow(2)

    # Normalize
    if do_normalize:
        for sens in sensitivities:
            sens = (sens-sens.mean())/sens.max()
    
    # Numerical considerations
    if do_numerical:
        for sens in sensitivities:
            # Add small number so as not to divide by extremely small numbers
            sens[sens == 0] = 10**-5
            # Clip sensitivities at a large constant value
            sens[sens > 10**5] = 10**5
            # Set any sensitivities of zero to one to avoid dividing by zero
            # param.grad.data[param.grad.data == 0] = 1

    # Set gradients
    for pid, param in enumerate(parent_model.parameters()):
        param.grad.data = sensitivities[pid].clone()
        #param.grad.data = torch.ones(sensitivities[pid].size())

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
    parent_model.zero_grad()

    # Shape returns and get rank of unperturbed model
    shaped_returns = fitness_shaping(returns)

    # Preallocate list with gradients
    gradients = []
    for param in parent_model.parameters():
        gradients.append(torch.zeros(param.data.size()))

    # Compute gradients
    # - ES strategy
    for i in range(args.agents):
        # Set random seed, get antithetic multiplier and reward
        np.random.seed(random_seeds[i])
        multiplier = -1 if is_anti_list[i] else 1
        reward = shaped_returns[i]
        for layer, param in enumerate(parent_model.parameters()):
            eps = np.random.normal(0, 1, param.data.size())
            grad = 1/(args.agents*args.sigma) * (reward*multiplier*eps)
            gradients[layer] += torch.from_numpy(grad).float()

    # Set gradients
    for layer, param in enumerate(parent_model.parameters()):
        param.grad.data = - gradients[layer]

    return parent_model


# TODO: Examine possibility of reusing pool of workers 
#       - p = Pool(args.agents)
#       - for _ in range(args.max_generations):
#       -   for j in range(int(args.agents/2)):
#       -       inputs.append((args, perturbed_model, seed, return_queue, env, is_negative))
#       -   p.imap_unordered(eval_fun, args=inputs) 
def train_loop(args, parent_model, env, eval_fun, optimizer, lr_scheduler, chkpt_dir, stats=None):
    # Initialize dict for saving statistics
    if stats is None:
        stat_names = ['generations', 'episodes', 'observations', 'walltimes',
                      'reward_avg', 'reward_var', 'reward_max', 'reward_min',
                      'reward_unp', 'unp_rank', 'sigma', 'lr']
        stats = {}
        for n in stat_names:
            stats[n] = []
    
    # Initialize return queue for multiprocessing
    return_queue = mp.Queue()
    
    # Evaluate parent model
    eval_fun(args, parent_model, 'dummy_seed', return_queue, env, 'dummy_neg', collect_inputs=True)
    unperturbed_out = return_queue.get()
    max_unperturbed_return = 0
    
    # Start training loop
    n_episodes = 0
    n_generation = 0
    n_observations = 0
    last_checkpoint_time = time.time()
    start_time = time.time()
    for n_generation in range(args.max_generations):
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
        
        # Compute gradients, update parameters and learning rate
        stats['lr'].append(get_lr(optimizer))
        compute_gradients(args, parent_model, returns, seeds, is_anti_list)
        optimizer.step()
        if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            lr_scheduler.step(unperturbed_out['return'])
        else:
            lr_scheduler.step()

        # Compute rank of unperturbed model
        rank_diag, rank = unperturbed_rank(returns, unperturbed_out['return'])
        
        # Keep track of best model
        if unperturbed_out['return'] >= max_unperturbed_return:
            best_model_state_dict = parent_model.state_dict()
            best_optimizer_state_dict = optimizer.state_dict()
            # TODO: Also save stats in "best" version
            max_unperturbed_return = unperturbed_out['return']

        # Store statistics
        n_episodes += len(returns)
        n_observations += sum(i_observations)
        stats['generations'].append(n_generation)
        stats['episodes'].append(n_episodes)
        stats['observations'].append(n_observations)
        stats['walltimes'].append(time.time() - start_time)
        stats['reward_avg'].append(np.mean(returns))
        stats['reward_var'].append(np.var(returns))
        stats['reward_max'].append(np.max(returns))
        stats['reward_min'].append(np.min(returns))
        stats['reward_unp'].append(unperturbed_out['return'])
        stats['unp_rank'].append(rank)
        stats['sigma'].append(args.sigma)
        
        if hasattr(args, 'variable_ep_len') and args.variable_ep_len:
            args.max_episode_length = int(5*max(num_frames))

        # Plot
        if last_checkpoint_time < time.time() - 10:
            # NOTE: Possible x-axis are: generations, episodes, observations, time
            fig = plt.figure()
            pltUnp, = plt.plot(stats['episodes'], stats['reward_unp'], label='parent')
            pltAvg, = plt.plot(stats['episodes'], stats['reward_avg'], label='average')
            pltMax, = plt.plot(stats['episodes'], stats['reward_max'], label='max')
            pltMin, = plt.plot(stats['episodes'], stats['reward_min'], label='min')
            plt.ylabel('rewards')
            plt.xlabel('episodes')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/rew_eps.pdf')
            plt.close(fig)

            fig = plt.figure()
            walltimes_plot = np.array(stats['walltimes'])/60
            pltUnp, = plt.plot(walltimes_plot, stats['reward_unp'], label='parent')
            pltAvg, = plt.plot(walltimes_plot, stats['reward_avg'], label='average')
            pltMax, = plt.plot(walltimes_plot, stats['reward_max'], label='max')
            pltMin, = plt.plot(walltimes_plot, stats['reward_min'], label='min')
            plt.ylabel('rewards')
            plt.xlabel('time [min]')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/rew_tim.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltUnp, = plt.plot(stats['observations'], stats['reward_unp'], label='parent')
            pltAvg, = plt.plot(stats['observations'], stats['reward_avg'], label='average')
            pltMax, = plt.plot(stats['observations'], stats['reward_max'], label='max')
            pltMin, = plt.plot(stats['observations'], stats['reward_min'], label='min')
            plt.ylabel('rewards')
            plt.xlabel('observations')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/rew_obs.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltUnp, = plt.plot(stats['generations'], stats['reward_unp'], label='parent')
            pltAvg, = plt.plot(stats['generations'], stats['reward_avg'], label='average')
            pltMax, = plt.plot(stats['generations'], stats['reward_max'], label='max')
            pltMin, = plt.plot(stats['generations'], stats['reward_min'], label='min')
            plt.ylabel('rewards')
            plt.xlabel('generations')
            plt.legend(handles=[pltUnp, pltAvg, pltMax,pltMin])
            fig.savefig(chkpt_dir+'/rew_gen.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltVar, = plt.plot(stats['generations'], stats['reward_var'], label='reward variance')
            plt.ylabel('reward variance')
            plt.xlabel('generations')
            plt.legend(handles=[pltVar])
            fig.savefig(chkpt_dir+'/rewvar_gen.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltVar, = plt.plot(stats['generations'], stats['unp_rank'], label='unperturbed rank')
            plt.ylabel('unperturbed rank')
            plt.xlabel('generations')
            plt.legend(handles=[pltVar])
            fig.savefig(chkpt_dir+'/unprank_gen.pdf')
            plt.close(fig)

            fig = plt.figure()
            pltVar, = plt.plot(stats['generations'], stats['sigma'], label='sigma')
            plt.ylabel('sigma')
            plt.xlabel('generations')
            plt.legend(handles=[pltVar])
            fig.savefig(chkpt_dir+'/sigma_gen.pdf')
            plt.close(fig)
            # TODO: Plot of time used
            # DONE: Plot of reward variance
            # TODO: Plot of smoothed reward
            # DONE: Plot unperturbed rank
            # DONE: Plot of sigma

        # Save checkpoint (every 5 minutes)
        if last_checkpoint_time < time.time() - 10:
            # Save latest model and optimizer state
            torch.save(parent_model.state_dict(), os.path.join(chkpt_dir, 'model_state_dict.pth'))
            torch.save(optimizer.state_dict(), os.path.join(chkpt_dir, 'optimizer_state_dict.pth'))
            # torch.save(lr_scheduler.state_dict(), os.path.join(chkpt_dir, 'lr_scheduler_state_dict.pth'))
            # Save best model
            torch.save(best_model_state_dict, os.path.join(chkpt_dir, 'best_model_state_dict.pth'))
            torch.save(best_optimizer_state_dict, os.path.join(chkpt_dir, 'best_optimizer_state_dict.pth'))
            # Currently, learning rate scheduler has no state_dict and cannot be saved. It can be restored
            # by setting lr_scheduler.last_epoch = last generation index.
            with open(os.path.join(chkpt_dir, 'stats.pkl'), 'wb') as filename:
                pickle.dump(stats, filename, pickle.HIGHEST_PROTOCOL)
            last_checkpoint_time = time.time()

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
                (stats['episodes'][-1], stats['reward_avg'][-1], stats['reward_var'][-1], stats['reward_max'][-1],
                stats['reward_min'][-1], args.agents, stats['sigma'][-1], stats['lr'][-1],
                unperturbed_out['return'], rank_diag, args.optimizer))
            print("Worker time: {}".format(workers_end_time - workers_start_time))
            print("Loop time: {}".format(time.time() - loop_start_time))
            print()

        
        