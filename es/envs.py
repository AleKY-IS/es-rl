from __future__ import absolute_import, division, print_function

import collections
import pkgutil

import atari_py
import gym
import IPython
import numpy as np
import torch
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

import cv2


# TODO: To circumvent universe, make rescale class for Atari, make normalize class wrapper
# TODO: Make create_env function for atari, classical control, mujoco etc.
def create_gym_environment(env_id, **kwargs):
    # spec = gym.spec(env_id)
    # get_gym_submodules_and_environments()
    # IPython.embed()
    # env_name, env_version = env_id.split('-')

    # envall = gym.envs.registry.all()

    # atari_games = atari_py.list_games()

    env = gym.make(env_id)
    if env_id in ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v0']:
        pass
    elif env_id in ['BipedalWalker-v2', 'LunarLander-v2']:
        pass
    elif env_id in ['Humanoid-v2']:
        pass
        # env = NormalizedEnv()
    else:
        env = Vectorize(env)
        env = AtariPreProcessorMnih2015(env)
        env = Unvectorize(env)
        # env = Vectorize(env)
        # env = AtariRescale(env, square_size=kwargs.get('square_size', 42))
        # env = Unvectorize(env)
    return env

def get_gym_submodules_and_environments():
    atari_games = atari_py.list_games()

    print('Searching gym.envs for submodules:')
    environments =  gym.envs.registry.all()

    for importer, modname, ispkg in pkgutil.iter_modules(gym.envs.__path__):
        print('  Found submodule {} (Package: {})'.format(modname, ispkg))
        try:
            m = importer.find_module(modname).load_module(modname)
        except gym.error.DependencyNotInstalled:
            pass
        if ispkg:
            for importer, modname, ispkg in pkgutil.iter_modules(getattr(gym.envs, modname).__path__):
                print('    Found environment {}'.format(modname))


# Taken from https://github.com/ikostrikov/pytorch-a3c
class NormalizedEnv(vectorized.ObservationWrapper):
    """
    Environment wrapper that maintains an estimate of the mean and standard deviation 
    for each observation channel and uses them to normalize the environment.
    """
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.max_episode_length = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.max_episode_length += 1
            self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        denom = (1 - pow(self.alpha, self.max_episode_length))
        unbiased_mean = self.state_mean / denom
        unbiased_std = self.state_std / denom

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8)
                for observation in observation_n]


class AtariPreProcessorMnih2015(vectorized.ObservationWrapper):
    """
    First, to encode a single frame we take the maximum value for each pixel 
    colour value over the frame being encoded and the previous frame. 
    This was necessary to remove flickering that is present in games where 
    some objects appear only in even frames while other objects appear only 
    in odd frames, an artefact caused by the limited number of sprites Atari 
    2600 can display at once. 
    Second, we then extract the Y channel, also known as luminance, from the 
    RGB frame and rescale it to 84 × 84. The function from algorithm 1 described 
    below applies this preprocessing to the m most recent frames and stacks them 
    to produce the input to the Q-function, in which m = 4, although the algorithm 
    is robust to different values of m (for example, 3 or 5).
    """

    def __init__(self, env=None):
        super(AtariPreProcessorMnih2015, self).__init__(env)
        self.n_previous_frames = 4
        self.frame_size = (84, 84)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.n_previous_frames, *self.frame_size))
        # Ring buffer of previous frames
        self.previous_frames = np.zeros(self.observation_space.shape)  # ollections.deque(maxlen=self.n_previous_frames)
        self.frame_counter = 0
    
    def _observation(self, frame):
        assert len(frame) == 1
        frame = frame[0]
        # import matplotlib.pyplot as plt
        # f = plt.figure()
        # plt.imshow(frame)
        # f.savefig('_1original.pdf')
        # Frame resizing
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)  # [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        # f = plt.figure()
        # plt.imshow(frame)
        # f.savefig('_2resized.pdf')
        # Grey scale conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # f = plt.figure()
        # plt.imshow(frame)
        # f.savefig('_3grey.pdf')
        # Fix odd-even flickering
        frame = np.maximum(frame, self.previous_frames[0, ...])
        # f = plt.figure()
        # plt.imshow(frame)
        # f.savefig('_4fixed.pdf')
        # Shift previous images by one and save new frame
        self.previous_frames = np.roll(self.previous_frames, shift=1, axis=0)
        self.previous_frames[0, ...] = frame
        return [self.previous_frames]

        # for interpol in [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]:
        #     t = time.time()
        #     for i in range(100000):
        #         frame_r = cv2.resize(frame, self.frame_size, interpolation=interpol)  # cv.INTER_LINEAR, INTER_AREA, INTER_NEAREST, INTER_CUBIC, INTER_LANCZOS4
        #     print(time.time() - t)
        #     f = plt.figure()
        #     plt.imshow(frame_r)
        #     f.savefig('_2resized_' + str(interpol) + '.pdf')
        

class AtariRescale(vectorized.ObservationWrapper):
    """
    Environment wrapper that resizes an Atari environment frame observation
    to the size defined by `square_size` pixels
    """

    def __init__(self, env=None, square_size=42):
        super(AtariRescale, self).__init__(env)
        self.square_size = square_size
        self.observation_space = Box(0.0, 1.0, [1, square_size, square_size])

    def _observation(self, observation_n):
        return [self._process_frame(observation) for observation in observation_n]

    def _process_frame(self, frame):
        frame = frame[34:34 + 160, :160]
        # Resize by half, then down to anything smaller than half (essentially mipmapping)
        # if wanted. If we resize directly we lose pixels that, when mapped to smaller sizes 
        # than 80 x 80, aren't close enough to the pixel boundary.
        if self.square_size < 80:
            frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (self.square_size, self.square_size))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [1, self.square_size, self.square_size])
        return frame


# class AtariProcessor(Processor):
#     """From Keras-RL
#     https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
#     """
#     def process_observation(self, observation):
#         assert observation.ndim == 3  # (height, width, channel)
#         img = Image.fromarray(observation)
#         img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
#         processed_observation = np.array(img)
#         assert processed_observation.shape == INPUT_SHAPE
#         return processed_observation.astype('uint8')  # saves storage in experience memory

#     def process_state_batch(self, batch):
#         # We could perform this processing step in `process_observation`. In this case, however,
#         # we would need to store a `float32` array instead, which is 4x more memory intensive than
#         # an `uint8` array. This matters if we store 1M observations.
#         processed_batch = batch.astype('float32') / 255.
#         return processed_batch

#     def process_reward(self, reward):
#         return np.clip(reward, -1., 1.)


def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)


class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log
