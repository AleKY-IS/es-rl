from __future__ import absolute_import, division, print_function

import math
import os

import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.legacy.optim as legacyOptim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable


def render_env(args, model, env):
    """
    Renders the learned model on the environment for testing.
    """
    try:
        while True:
            # Reset environment
            state = env.reset()
            state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
            this_model_return = 0
            this_model_num_frames = 0
            done = False
            # Rollout
            while not done and this_model_num_frames < args.max_episode_length:
                # Choose action
                actions = model(state)
                action = actions.max(1)[1].data.numpy()
                # Step
                state, reward, done, _ = env.step(action[0])
                this_model_return += reward
                this_model_num_frames += 1
                # Cast state
                state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
                env.render()
            print('Reward: %f' % this_model_return)
    except KeyboardInterrupt:
        print("\nEnded test session by keyboard interrupt")


# TODO: Take this method as input such that it can be defined depending on the environment 
#       This one here is for Atari environments
def do_rollouts(args, models, random_seeds, return_queue, env, are_negative):
    """
    Do rollouts of policy defined by model in given environment. 
    Has support for multiple models per thread, but it is inefficient.
    """
    all_returns = []
    all_num_frames = []
    for model in models:
        # Reset environment
        state = env.reset()
        state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)
        this_model_return = 0
        this_model_num_frames = 0
        done = False
        # Rollout
        while not done and this_model_num_frames < args.max_episode_length:
            # Choose action
            actions = model(state)
            action = actions.max(1)[1].data.numpy()
            # Step
            state, reward, done, _ = env.step(action[0])
            this_model_return += reward
            this_model_num_frames += 1
            # Cast state
            state = Variable(torch.from_numpy(state).float(), volatile=True).unsqueeze(0)

        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames, are_negative))


def perturb_model(args, model, random_seed, env):
    """
    Modifies the given model with a pertubation of its parameters,
    as well as the mirrored perturbation, and returns both perturbed
    models.
    """
    # Get model class and instantiate two new models as copies of parent
    model_class = type(model)
    model1 = model_class(env.observation_space, env.action_space)
    model2 = model_class(env.observation_space, env.action_space)
    model1.load_state_dict(model.state_dict())
    model2.load_state_dict(model.state_dict())
    np.random.seed(random_seed)
    # Permute all weights of each model by isotropic Gaussian noise
    for (k, v), (anti_k, anti_v) in zip(model1.es_params(),
                                        model2.es_params()):
        eps = np.random.normal(0, 1, v.size())
        v += torch.from_numpy(args.sigma*eps).float()
        anti_v += torch.from_numpy(args.sigma*-eps).float()
    return [model1, model2]


def generate_seeds_and_models(args, synced_model, env):
    """
    Returns a seed and 2 perturbed models
    """
    np.random.seed()
    random_seed = np.random.randint(2**30)
    two_models = perturb_model(args, synced_model, random_seed, env)
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


def unperturbed_rank(returns, unperturbed_results):
    nth_place = 1
    for r in returns:
        if r > unperturbed_results:
            nth_place += 1
    rank_diag = ('%d out of %d (1 means gradient '
                    'is uninformative)' % (nth_place,
                                            len(returns) + 1))
    return rank_diag, nth_place


optimConfig = []
averageReward = []
maxReward = []
minReward = []
episodeCounter = []


# TODO: Move plotting, printing and saving into train loop
# TODO: Fix saving to include episode number etc
# TODO: Fix saving to be into file's directory
# TODO: (Save only the model if it is better than the previous one)
# TODO: Make checkpoint based on time (every 1 minute e.g.)
def gradient_update(args, synced_model, returns, random_seeds, neg_list,
                    num_eps, num_frames, chkpt_dir, unperturbed_results):
    batch_size = len(returns)
    assert batch_size == args.n
    assert len(random_seeds) == batch_size
    # Shape returns and get rank of unperturbed model
    shaped_returns = fitness_shaping(returns)
    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)

    averageReward.append(np.mean(returns))
    episodeCounter.append(num_eps)
    maxReward.append(max(returns))
    minReward.append(min(returns))

    # Print (TODO: Move out of this function)
    if not args.silent:
        print('Episode num: %d\n'
              'Average reward: %f\n'
              'Variance in rewards: %f\n'
              'Max reward: %f\n'
              'Min reward: %f\n'
              'Batch size: %d\n'
              'Max episode length: %d\n'
              'Sigma: %f\n'
              'Learning rate: %f\n'
              'Total num frames seen: %d\n'
              'Unperturbed reward: %f\n'
              'Unperturbed rank: %s\n' 
              'Using Adam: %r\n\n' %
              (num_eps, np.mean(returns), np.var(returns), max(returns),
               min(returns), batch_size,
               args.max_episode_length, args.sigma, args.lr, num_frames,
               unperturbed_results, rank_diag, args.useAdam))

    # Plot (TODO: Move out)
    if num_eps % (40*20) == 0:
        fig = plt.figure()
        pltAvg, = plt.plot(episodeCounter, averageReward, label='average')
        pltMax, = plt.plot(episodeCounter, maxReward,  label='max')
        pltMin, = plt.plot(episodeCounter, minReward,  label='min')
        plt.ylabel('rewards')
        plt.xlabel('episode num')
        plt.legend(handles=[pltAvg, pltMax,pltMin])
        fig.savefig(chkpt_dir+'/graph.pdf')
        plt.close(fig)
        print('Updated plot')

    # TODO: Add save checkpoint including episode # etc.

    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    # TODO: Move shared code out of branch
    if args.useAdam:
        globalGrads = None
        for i in range(args.n):
            np.random.seed(random_seeds[i])
            multiplier = -1 if neg_list[i] else 1
            reward = shaped_returns[i]

            localGrads = []
            idx = 0
            for k, v in synced_model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad = torch.from_numpy((args.n*args.sigma) * (reward*multiplier*eps)).float()

                localGrads.append(grad)
                
                if len(optimConfig) == idx:
                    optimConfig.append({'learningRate': args.lr})
                idx = idx + 1

            if globalGrads is None:
                globalGrads = localGrads
            else:
                for i in range(len(globalGrads)):
                    globalGrads[i] = torch.add(globalGrads[i], localGrads[i])

        idx = 0
        for k, v in synced_model.es_params():
            r, _ = legacyOptim.adam(lambda x:  (1, -globalGrads[idx]), v, optimConfig[idx])
            v.copy_(r)
            idx = idx + 1
    else:
        # For each model, generate the same random numbers as we did
        # before, and update parameters. We apply weight decay once.
        for i in range(args.n):
            np.random.seed(random_seeds[i])
            multiplier = -1 if neg_list[i] else 1
            reward = shaped_returns[i]
            for k, v in synced_model.es_params():
                eps = np.random.normal(0, 1, v.size())
                grad = torch.from_numpy((args.n*args.sigma) * (reward*multiplier*eps)).float()
                v += args.lr * grad
        args.lr *= args.lr_decay

    # TODO: Move out
    torch.save(synced_model.state_dict(), os.path.join(chkpt_dir, 'latest.pth'))
    return synced_model


def train_loop(args, synced_model, env, chkpt_dir):
    def flatten(raw_results, index):
        notflat_results = [result[index] for result in raw_results]
        return [item for sublist in notflat_results for item in sublist]
    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    for _ in range(args.max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        all_seeds, all_models = [], []

        # Generate a perturbation and its antithesis
        for j in range(int(args.n/2)):
            random_seed, two_models = generate_seeds_and_models(args, synced_model, env)
            # Add twice because we get two models with the same seed
            all_seeds.append(random_seed)
            all_seeds.append(random_seed)
            all_models.extend(two_models)
        assert len(all_seeds) == len(all_models)

        # Keep track of which perturbations were positive and negative
        # Start with negative true because pop() makes us go backwards
        is_negative = True
        # Add all peturbed models to the queue
        while all_models:
            perturbed_model = all_models.pop()
            seed = all_seeds.pop()
            inputs = (args, [perturbed_model], [seed], return_queue, env, [is_negative])
            p = mp.Process(target=do_rollouts, args=inputs)
            p.start()
            processes.append(p)
            is_negative = not is_negative
        assert len(all_seeds) == 0

        # Evaluate the unperturbed model as well
        p = mp.Process(target=do_rollouts, args=(args, [synced_model], ['dummy_seed'], return_queue, env, ['dummy_neg']))
        p.start()
        processes.append(p)

        # Get results of started processes
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames, neg_list = [flatten(raw_results, index) for index in [0, 1, 2, 3]]

        # Separate the unperturbed results from the perturbed results
        _ = unperturbed_index = seeds.index('dummy_seed')
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        _ = num_frames.pop(unperturbed_index)
        _ = neg_list.pop(unperturbed_index)

        # Update gradient
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        synced_model = gradient_update(args, synced_model, results, seeds,
                                       neg_list, num_eps, total_num_frames,
                                       chkpt_dir, unperturbed_results)

        if args.variable_ep_len:
            args.max_episode_length = int(5*max(num_frames))

        # Save checkpoint and print (TODO move stuff to here) 
