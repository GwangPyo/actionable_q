import numpy as np
import pandas as pd
from stable_baselines3.common.utils import get_device
import torch as th
from functools import wraps
from typing import NamedTuple, Dict


class ActionableReplayData(NamedTuple):
    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: Dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    goal: th.Tensor


def collate_fn(method):
    @wraps(method)
    def _imple(self, *args, **kwargs):
        ret = method(self, *args, **kwargs)
        torch_dict = {k: th.from_numpy(ret[k]).float().to(self.device) for k in ret.keys()}
        torch_dict['observations'] = {'observations': torch_dict['observations'], 'goal': torch_dict['goal']}
        torch_dict['next_observations'] = {'observations': torch_dict['next_observations'], 'goal': torch_dict['goal']}
        torch_dict['rewards'] = torch_dict['rewards'].reshape(-1, 1)
        torch_dict['dones'] = torch_dict['dones'].reshape(-1, 1)
        return ActionableReplayData(**torch_dict)

    return _imple


class ActionableDataset(object):
    def __init__(self, filename: str, device='auto'):
        dictionary = dict(np.load(filename))

        self.observations = dictionary["observations"]
        self.next_observation = dictionary["next_observations"]
        self.actions = dictionary["actions"]
        self.dones = dictionary["terminals"]
        keys = ("epilen", "episode_id")
        dictionary = {k: list(dictionary[k]) for k in keys}

        self._len = len(dictionary['episode_id'])
        self.data = pd.DataFrame.from_dict(dictionary)
        self.min_epilen = self.data['epilen'].min()
        self.num_episode = int(self.data['episode_id'].max())

        if device == 'auto':
            self.device = get_device()
        else:
            self.device = device

    def select_episode(self, index):
        return self.data[self.data.episode_id == index]

    def fetch_trajectory(self, size):
        """
        Fetch partition of trajectory given size
        The size is clamped to epsiode length - 1
        :param size: size of trajectory to fetch
        :return: dictionary of trajectories
        Note that the reward is "zeros" except the "goal" condition
        """
        idx = np.random.randint(low=0, high=self.num_episode)
        # random choice of episode
        episode = self.select_episode(idx)
        start = episode.index[0]

        epilen = episode['epilen'].iloc[[0]].values.item()

        size = min(epilen - 1, size)
        cut_start = np.random.randint(low=0, high=epilen - size) + start
        cut_end = size + cut_start

        observations = self.observations[cut_start:cut_end]
        actions = self.actions[cut_start: cut_end]
        next_observations = self.next_observation[cut_start: cut_end]
        goal = np.repeat(next_observations[-1][None], (cut_end - cut_start), axis=0)

        rewards = np.zeros(shape=(cut_end - cut_start, ), dtype=np.float32)
        rewards[-1] = 1
        terminals = np.logical_or(self.dones[cut_start: cut_end], rewards)

        return {"observations": observations,
                "actions": actions,
                "next_observations": next_observations,
                "goal": goal,
                "rewards": rewards,
                "dones": terminals}

    @collate_fn
    def fetch(self, n_parallel=4, ):

        size = np.random.randint(low=2, high=self.min_epilen, size=(n_parallel, ))

        ret = {"observations": [], "actions": [], "next_observations": [],
               "goal": [], "rewards": [], "dones": []}

        fetched = [self.fetch_trajectory(s) for s in size]
        for f in fetched:
            for k in ret.keys():
                ret[k].append(np.asarray(f[k]))
        ret = {k: np.concatenate(ret[k], axis=0) for k in ret.keys()}
        return ret
