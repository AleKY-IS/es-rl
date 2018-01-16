from __future__ import absolute_import, division, print_function

import numpy as np

import gym
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

import cv2

import IPython

# TODO: To circumvent universe, make rescale class for Atari, make normalize class wrapper
# TODO: Make create_env function for atari, classical control, mujoco etc.
def create_env(env_id, **kwargs):
    spec = gym.spec(env_id)
    return env

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id, square_size=42):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale(env, square_size=square_size)
    #env = DiagnosticsInfo(env)
    env = NormalizedEnv(env)
    env = Unvectorize(env)
    return env


def create_classical_control_env(env_id):
    env = gym.make(env_id)
    return env


def create_mujoco_env(env_id):
    env = gym.make(env_id)
    return env


class AtariRescale(vectorized.ObservationWrapper):

    def __init__(self, env=None, square_size=42):
        super(AtariRescale, self).__init__(env)
        self.square_size = square_size
        self.observation_space = Box(0.0, 1.0, [1, square_size, square_size])

    def _observation(self, observation_n):
        return [self._process_frame(observation) for observation in observation_n]

    def _process_frame(self, frame):
        frame = frame[34:34 + 160, :160]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        if self.square_size < 80:
            frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (self.square_size, self.square_size))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [1, self.square_size, self.square_size])
        return frame


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
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        denom = (1 - pow(self.alpha, self.max_episode_length))
        unbiased_mean = self.state_mean / denom
        unbiased_std = self.state_std / denom

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8)
                for observation in observation_n]
