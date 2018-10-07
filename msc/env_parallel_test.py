import copy
import multiprocessing as mp
import time

import gym
import IPython

max_episode_length = 100000

def rollout(env):
    state = env.reset()
    this_model_num_steps = 0
    retrn = 0
    done = False
    # Rollout
    while not done and this_model_num_steps < max_episode_length:
        # Choose action
        action = env.action_space.sample()
        # Step
        state, reward, done, _ = env.step(action)
        retrn += reward
        this_model_num_steps += 1
    return retrn, this_model_num_steps

def eval_wrap(env):
    e = copy.deepcopy(env)
    r = rollout(e)
    return r

pool = mp.Pool()
env = gym.make('Freeway-v0')
# env = gym.make('BipedalWalker-v2')
n = 30

t_parallel1 = time.time()
out = []
for i in range(n):
    out.append(pool.apply_async(rollout, (env,)))
for o in out:
    o.get()
t_parallel1 = time.time() - t_parallel1


t_parallel2 = time.time()
out = []
for i in range(n):
    out.append(pool.apply_async(eval_wrap, (env,)))
for o in out:
    o.get()
t_parallel2 = time.time() - t_parallel2


t_parallel3 = time.time()
out = []
envs = [copy.deepcopy(env) for i in range(n)]
for e in envs:
    out.append(pool.apply_async(rollout, (e,)))
for o in out:
    o.get()
t_parallel3 = time.time() - t_parallel3


t_parallel4 = time.time()
out = pool.map_async(rollout, [env]*n)
out.get()
t_parallel4 = time.time() - t_parallel4


t_sequential = time.time()
for i in range(n):
    rollout(env)
t_sequential = time.time() - t_sequential

print(t_parallel1)
print(t_parallel2)
print(t_parallel3)
print(t_sequential)


# No need for copying environment
# Doing so only slows down code - more so if done ahead of time (sequentially)
# Mapping is slightly faster than applying asynchronously in a for loop
