import torch as th
from torch import nn, Tensor
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Sequence, Dict


class Flatter(nn.ModuleDict):
    """
    Flatten all dictionary observation
    """
    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__(modules)
        keys = list(modules.keys())
        keys.sort()
        self.keys = keys

    def forward(self, inputs: Dict[str, Tensor]):
        forwarded = [self[k](inputs[k]) for k in self.keys]
        return th.cat(forwarded, dim=-1)


class GoalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 features_dim: int = 64,
                 net_arch: Sequence[int] = (64, 64)):
        assert isinstance(observation_space, spaces.Dict)
        obs_space = {}

        for k in observation_space:
            obs_space[k] = get_flattened_obs_dim(observation_space[k])

        super().__init__(observation_space,
                         features_dim=features_dim)

        self.summation_dim = sum(obs_space.values())
        self.flattens = Flatter({
            k: FlattenExtractor(observation_space[k]) for k in observation_space.keys()
        })
        self.extractors = nn.Sequential(*create_mlp(self.summation_dim, output_dim=self.features_dim,
                                                    net_arch=list(net_arch)))

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        flatten_cat_observations = self.flattens(observations)
        return self.extractors(flatten_cat_observations)
    