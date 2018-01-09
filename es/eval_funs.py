import torch
from torch.autograd import Variable
import IPython


def gym_rollout(args, models, random_seeds, return_queue, env, is_antithetic):
    """
    Do rollouts of policy defined by model in given environment. 
    Has support for multiple models per thread, but it is inefficient.
    """
    all_returns = []
    all_num_frames = []
    for model in models:
        # Reset environment
        state = env.reset()
        state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
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
            state = Variable(torch.from_numpy(state).float(), requires_grad=True).unsqueeze(0)
            
            print(state.requires_grad)
            print(state.grad)
            state.backward(torch.ones(state.shape))
            print(state.grad)

        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames, is_antithetic))


def gym_render(args, model, env):
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


def gym_rollout_nogradients(args, models, random_seeds, return_queue, env, is_antithetic):
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

    return_queue.put((random_seeds, all_returns, all_num_frames, is_antithetic))




def supervised_fitness(args, models, random_seeds, return_queue, env, is_antithetic):



    return_queue.put((random_seeds, all_returns, all_num_frames, is_antithetic))